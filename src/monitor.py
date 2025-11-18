import pandas as pd
import os
import logging
import datetime
import tempfile
from typing import Optional
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

import mlflow


logger = logging.getLogger(__name__)

# Load environment variables from .env located at project root
# This prevents committing sensitive values to version control
load_dotenv()

# Read config from env with sensible defaults
REFERENCE_PATH = os.getenv("REFERENCE_DATA_PATH")
CURRENT_PATH = os.getenv("NEW_DATA_PATH")
DRIFT_WARNING_THRESHOLD = float(os.getenv("DRIFT_WARNING_THRESHOLD", "0.2"))
DRIFT_CRITICAL_THRESHOLD = float(os.getenv("DRIFT_CRITICAL_THRESHOLD", "0.3"))
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", "300"))
LOG_DRIFT_TO_MLFLOW = os.getenv("LOG_DRIFT_TO_MLFLOW", "true").lower() in ("1", "true", "yes")


class DataDriftMonitor:
    """
    Class to monitor data drift between reference and current datasets using Evidently.
    Compatible with Evidently 0.7.x API.
    """
    
    def __init__(self,
        reference_path: Optional[str] = None,
        current_path: Optional[str] = None):
        self.reference_path = reference_path or REFERENCE_PATH
        self.current_path = current_path or CURRENT_PATH
        self.last_status = {
            "timestamp": None,
            "drift_score": None,
            "level": "unknown",
            "recommended_action": None
        }

    def _load_data(self):
        """Load reference and current datasets from CSV files."""
        ref = pd.read_csv(self.reference_path)
        cur = pd.read_csv(self.current_path)
        return ref, cur

    def _evaluate(self, drift_score: float):
        """Evaluate drift level and recommend actions based on thresholds."""
        if drift_score > DRIFT_CRITICAL_THRESHOLD:
            return "critical", "Retrain modelo + revisión completa del pipeline de features."
        if drift_score > DRIFT_WARNING_THRESHOLD:
            return "warning", "Revisar cambios recientes en la distribución de features."
        return "normal", "Sin acción requerida."

    def _log_to_mlflow(self, drift_score: float, level: str, report_html_path: Optional[str]):
        """Log drift metrics and reports to MLflow."""
        if not LOG_DRIFT_TO_MLFLOW:
            return
        try:
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "data_drift_monitoring"))
            with mlflow.start_run(run_name=f"drift_check_{datetime.datetime.now(datetime.timezone.utc).isoformat()}"):
                mlflow.log_metric("data_drift_score", float(drift_score))
                mlflow.log_param("drift_level", level)
                if report_html_path and os.path.exists(report_html_path):
                    mlflow.log_artifact(report_html_path, artifact_path="drift_reports")
                logger.info("Logged drift to MLflow")
        except Exception as e:
            logger.exception(f"Failed to log drift to MLflow: {e}")

    def _build_evidently_report(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        """
        Build and run Evidently Report using the new API (0.7.x).
        
        The new API:
        - Report.run() returns an evaluation result object (not the Report itself)
        - Use .dict() or .json() on the result, not .as_dict()
        - Use .save_html() on the result object
        
        Returns:
            The evaluation result object from report.run()
        """
        report = Report(metrics=[
            DataDriftPreset()
        ])
        # Run returns the evaluation result object
        my_eval = report.run(reference_df, current_df)
        return my_eval

    def _extract_drift_score(self, my_eval) -> float:
        """Extract drift score for Evidently 0.7.x using 'DriftedColumnsCount' metric."""
        try:
            result = my_eval.dict()
            metrics = result.get("metrics", [])

            for metric in metrics:
                metric_id = metric.get("metric_id", "")
                
                # We look specifically for the dataset drift metric
                if metric_id.startswith("DriftedColumnsCount"):
                    value = metric.get("value", {})
                    # Extract the drift share (already provided by Evidently)
                    drift_share = value.get("share")
                    if drift_share is not None:
                        return float(drift_share)

            # If not found, return 0 with warning
            logger.warning("DriftedColumnsCount metric not found; returning 0")
            return 0.0

        except Exception as e:
            logger.exception(f"Error extracting drift score: {e}")
            return 0.0

    def get_report_html(self, reference_df: Optional[pd.DataFrame] = None, 
                       current_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate HTML report and return its contents as a string.
        
        This method creates a temporary HTML file and reads its contents.
        """
        if reference_df is None or current_df is None:
            reference_df, current_df = self._load_data()

        # Build and run report using new API
        my_eval = self._build_evidently_report(reference_df, current_df)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w') as f:
            temp_path = f.name
        
        # Save HTML report using new API - call save_html on the evaluation result
        my_eval.save_html(temp_path)
        
        # Read and return contents
        with open(temp_path, "r", encoding="utf-8") as fh:
            html = fh.read()
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return html

    def run_check(self, save_report_html: bool = True) -> dict:
        """
        Run a single data drift check using Evidently's new API (0.7.x).
        
        Returns:
            dict with keys: drift_score, level, recommended_action, report_path (optional)
        """
        try:
            ref, cur = self._load_data()
            
            # Build and run report using new API
            my_eval = self._build_evidently_report(ref, cur)
            
            # Extract drift score from evaluation result
            drift_score = self._extract_drift_score(my_eval)
            
            # Evaluate drift level
            level, action = self._evaluate(drift_score)

            # Save HTML report if requested
            report_path = None
            if save_report_html:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                    report_path = f.name
                
                # Save using new API - call save_html on the evaluation result
                my_eval.save_html(report_path)
                logger.info(f"Saved drift report to {report_path}")

            # Update last_status
            self.last_status = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "drift_score": float(drift_score),
                "level": level,
                "recommended_action": action,
                "report_path": report_path
            }

            # Send notifications on critical drift
            if level == "critical":
                text = (f"[CRITICAL] Data drift detected (score={drift_score:.3f}). "
                       f"Recommended action: {action}")
                print(text)

            # Log to MLflow
            self._log_to_mlflow(drift_score, level, report_path)

            logger.info(f"Drift run complete: {self.last_status}")
            return self.last_status

        except Exception as e:
            logger.exception(f"Data drift run failed: {e}")
            return {"error": str(e)}
        
def monitor():
     # Initialize monitor
    monitor = DataDriftMonitor(
        reference_path=str(REFERENCE_PATH),
        current_path=str(CURRENT_PATH)
    )
    
    # Run drift check
    logger.info("Running drift check...")
    result = monitor.run_check(save_report_html=True)
    
    # Check if there was an error
    if "error" in result:
        print("\n" + "="*60)
        print("ERROR OCCURRED:")
        print("="*60)
        print(f"Error: {result['error']}")
        print("="*60 + "\n")
        return result
    
    # Display results
    print("\n" + "-"*60)
    print("DRIFT CHECK RESULTS:")
    print("-"*60)
    print(f"Timestamp: {result.get('timestamp', 'N/A')}")
    print(f"Drift Score: {result.get('drift_score', 0.0):.4f}")
    print(f"Level: {result.get('level', 'unknown').upper()}")
    print(f"Recommended Action: {result.get('recommended_action', 'N/A')}")
    if result.get('report_path'):
        print(f"HTML Report: {result['report_path']}")
        print(f"Open with: file://{os.path.abspath(result['report_path'])}")
    print("-"*60 + "\n")
    
    return result

if __name__ == "__main__":
    monitor()