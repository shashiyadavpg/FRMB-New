import os
import sys

def run_step(step_name, command):
    print(f"\n[PIPELINE] Starting: {step_name}")
    ret = os.system(command)
    if ret != 0:
        print(f"[PIPELINE] Error in {step_name}. Exiting.")
        sys.exit(ret)
    print(f"[PIPELINE] Completed: {step_name}")

if __name__ == "__main__":
    # Assumes ingestion is done or skipped
    
    # 1. Consolidate Data
    run_step("Data Consolidation", "python src/data_processor.py")
    
    # 2. Feature Engineering
    run_step("Feature Engineering", "python src/features.py")
    
    # 3. Model Training
    run_step("Model Training", "python src/model_trainer.py")
    
    # 4. Report Generation
    run_step("Report Generation", "python src/report_generator.py")
    
    # 5. Visuals
    run_step("Visualization", "python src/visualizer.py")
    
    print("\n[PIPELINE] All steps finished successfully. Check 'outputs/' directory.")
