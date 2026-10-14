"""Internal stage runner; users run study.py."""
import argparse
from pipeline.config import load_config
from pipeline.endpoint_workflow import STAGES, execute_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--stage",choices=STAGES,required=True)
    parser.add_argument("--thermal-config",default="thermal_analysis.yaml")
    args = parser.parse_args()
    print(execute_stage(load_config(args.config),args.stage,args.thermal_config))
