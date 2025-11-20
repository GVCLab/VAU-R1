import os
import argparse
import numpy as np
import pandas as pd
import json
import re
import markdown
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
)


def time_to_seconds(time_str):
    if isinstance(time_str, (int, float)):
        return time_str

    lennormal_gth = len(time_str.split(":"))
    if len(time_str.split(".")) > 2:
        time_str = ".".join(time_str.split(".")[:2]) 
    float_state = True if len(time_str.split(".")) == 2 else False
    
    if lennormal_gth == 3:
        time_obj = datetime.strptime(time_str, f'%H:%M:%S' + (".%f" if float_state else ''))
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
    elif lennormal_gth == 2:
        time_obj = datetime.strptime(time_str, f'%M:%S' + (".%f" if float_state else ''))
        total_seconds = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
    elif lennormal_gth == 1:
        time_obj = datetime.strptime(time_str, f'%S' + (".%f" if float_state else ''))
        total_seconds = time_obj.second + time_obj.microsecond / 1_000_000
    return total_seconds


def parse_html(response: str) -> dict:
    """
    Extract the first html object or recognise the
    “No anomaly detected” message.

    response: "<answer>Anomaly</answer><glue>[0.0, 10.4]</glue>"

    Always returns:
        {"start time": value, "end time": value}
    """
    glue = re.search(r"<glue>(.*?)</glue>", response)
    if not glue:
        # check if contain "[0.5,1.2]"
        glue = re.search(r"\[(.*?)\]", response)
        if not glue:
            return {
                "start time": 0.0,
                "end time":   0.0,
            }

    time_str = glue.group(1)
    # remove all non-numeric characters except for : and ,
    time_str = re.sub(r"[^0-9:.,-]", "", time_str)
    if len(time_str.split(",")) != 2: 
        return {
            "start time": 0.0,
            "end time":   0.0,
        }
        
    st, ed = time_str.split(",")

    if ":" in st and ":" in ed:
        return {
            "start time": time_to_seconds(st),
            "end time":   time_to_seconds(ed),
        }

    return {
        "start time": float(st),
        "end time": float(ed),
    }


def _round(value, decimal_places=4):
    return round(value, decimal_places) * 100


class TemporalEvaluator:
    def __init__(self, args):
        self.test_gt_path = args.test_gt_path
        self.pred_path = args.pred_path
        self.save_path = args.save_path

        self.gt_df = self.load_gt()
        self.pred_df = self.parse_pred()

        
    def load_gt(self):
        try:
            df = pd.read_csv(self.test_gt_path)
            print(f"Loaded ground truth file, shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading ground truth file: {e}")
            return None

    def parse_pred(self):
        try:
            df = pd.read_csv(self.pred_path)
            duplicate_rows = df[df.duplicated(subset=["Video Name"], keep=False)]
            if not duplicate_rows.empty:
                print(f"Duplicate rows found in prediction file: {duplicate_rows}")
                # save the first duplicate and drop the rest
                df = df.drop_duplicates(subset=["Video Name"], keep="first")
            print(f"Loaded prediction file, shape: {df.shape}")
        except Exception as e:
            print(f"Error loading prediction file: {e}")
            return None

        # read each line and parse the temporal grounding to start and end time
        for index, row in df.iterrows():
            video_name = row["Video Name"]
            raw_answer = row["Temporal Grounding"]

            # if name w/o .mp4, add .mp4
            video_name = str(video_name).strip()
            if not video_name.endswith(".mp4"):
                df.loc[index, "Video Name"] = video_name + ".mp4"

            # Parse the temporal grounding 
            data = parse_html(raw_answer)
            df.loc[index, "start time"] = data["start time"]
            df.loc[index, "end time"] = data["end time"]

        return df

    def get_gt_label(self, video_name):
        gt_start = self.gt_df.loc[self.gt_df["Video Name"] == video_name, "Start Time"].values[0]
        gt_end = self.gt_df.loc[self.gt_df["Video Name"] == video_name, "End Time"].values[0]
        return (gt_start == 0.0) and (gt_end == 0.0)


    def eval_IoU(self, video_name):
        normal_gt_start = self.gt_df.loc[self.gt_df["Video Name"] == video_name, "Start Time"].values[0]
        normal_gt_end   = self.gt_df.loc[self.gt_df["Video Name"] == video_name, "End Time"].values[0]
        pred_start      = self.pred_df.loc[self.pred_df["Video Name"] == video_name, "start time"].values[0]
        pred_end        = self.pred_df.loc[self.pred_df["Video Name"] == video_name, "end time"].values[0]

        # “no-detection” fallback
        if pred_start == pred_end:
            return 0

        intersection = max(0, min(normal_gt_end, pred_end) - max(normal_gt_start, pred_start))
        union        = max(normal_gt_end, pred_end) - min(normal_gt_start, pred_start)
        iou = intersection / union if union > 0 else 0

        return iou

    def evaluate(self):
        if self.gt_df is None or self.pred_df is None:
            print("Error: Ground truth or prediction data is not loaded properly.")
            return

        iou_scores = []
        valid_total = 0
        recall_iou = {
            "r@0.3": 0,
            "r@0.5": 0,
            "r@0.7": 0,
        }

        for video_name in self.pred_df["Video Name"].values:
            # skip if video name is not in gt
            if video_name not in self.gt_df["Video Name"].values:
                # skip video name not in gt
                continue

            if self.get_gt_label(video_name): 
                # skip normal videos
                continue

            iou = self.eval_IoU(video_name)
            if iou != -1:
                iou_scores.append(iou) 
            if iou >= 0.3:
                recall_iou["r@0.3"] += 1
            if iou >= 0.5:
                recall_iou["r@0.5"] += 1
            if iou >= 0.7:
                recall_iou["r@0.7"] += 1
            valid_total += 1
        
        iou_scores = np.array(iou_scores)
        mean_iou = _round(iou_scores.mean())
        recall_iou["r@0.3"] = _round(recall_iou["r@0.3"] / len(iou_scores))
        recall_iou["r@0.5"] = _round(recall_iou["r@0.5"] / len(iou_scores))
        recall_iou["r@0.7"] = _round(recall_iou["r@0.7"] / len(iou_scores))

        # print the results
        print(f"Mean IoU: {mean_iou}%")
        print(f"Recall @0.3: {recall_iou['r@0.3']}%")
        print(f"Recall @0.5: {recall_iou['r@0.5']}%")
        print(f"Recall @0.7: {recall_iou['r@0.7']}%")
        print("-" * 40)

        # save the results to txt
        with open(self.save_path, "w") as f:
            f.write(f"Valid Predictions: {valid_total}\n") 
            f.write(f"Mean IoU: {mean_iou}%\n")
            f.write(f"Recall @0.3: {recall_iou['r@0.3']}%\n")
            f.write(f"Recall @0.5: {recall_iou['r@0.5']}%\n")
            f.write(f"Recall @0.7: {recall_iou['r@0.7']}%\n")
        print(f"Evaluation results saved to {self.save_path}\n", "-"*40, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate temporal grounding results")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the ground truth CSV file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the result CSV file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the evaluation results (default: pred_path with _eval suffix)")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.pred_path.replace(".csv", "_eval.txt")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    TemporalEvaluator(args).evaluate()