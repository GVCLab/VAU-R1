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

FIXED_OPTION = [
    "Abuse", "Agriculture", "Arrest", "Arson", "Assault", "Explosion",
    "Fighting", "Fire", "Food Safety", "Medical Abnormality", "Natural Hazard",
    "Normal", "Object_falling", "People_falling", "Robbery", "Shooting",
    "Stealing", "Traffic_accident", "Vandalism", "Water_incident",
]


def _normalize(text: str) -> str:
    """Lower-case and collapse runs of space/underscore to one underscore."""
    return re.sub(r"[\s_]+", "_", text.strip().lower())

def _build_option_pattern(opts) -> re.Pattern:
    """
    Matches any option, allowing arbitrary runs of spaces/underscores
    between its words and ignoring case.
    """
    parts = []
    for opt in opts:
        # "Traffic_accident"  ->  r"Traffic[_\s]*accident"
        words = map(re.escape, re.split(r"[_\s]+", opt))
        parts.append(r"(?:%s)" % r"[_\s]*".join(words))
    return re.compile(r"\b(" + "|".join(parts) + r")\b", flags=re.IGNORECASE)

_OPTION_PATTERN = _build_option_pattern(FIXED_OPTION)
_CANONICAL_BY_NORM = {_normalize(o): o for o in FIXED_OPTION}


def extract_options(response):
    seen = set()
    ordered_options = []

    # if response has <answer> tag, take the result for the tag
    if "<answer>" in response:
        # parse the response as html
        soup = BeautifulSoup(response, "html.parser")
        answer = soup.find("answer")
        if answer:
            response = answer.text

    for m in _OPTION_PATTERN.finditer(response):
        norm = _normalize(m.group(1))
        canonical = _CANONICAL_BY_NORM.get(norm)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered_options.append(canonical)
        
    return ordered_options


def _round(value, decimal_places=4):
    return round(value, decimal_places) * 100


class CLSEvaluator:
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
            # parse the answer
            df.at[index, "CLS answer"] = extract_options(row["CLS answer"])
        return df


    def evaluate(self):
        if self.gt_df is None or self.pred_df is None:
            print("Error: Ground truth or prediction data is not loaded properly.")
            return

        # calculate acc
        binary_acc, cls_acc = [], []
        valid_total = 0  # <---- NEW
        for index, row in self.gt_df.iterrows():
            video_name = str(row["Video Name"])
            if ".mp4" not in video_name:
                video_name = f"{video_name}.mp4"
            gt = row["Anomaly Class"]
            
            pred_row = self.pred_df[self.pred_df["Video Name"] == video_name]
            pred = pred_row["CLS answer"].values[0] if not pred_row.empty else []
            if len(pred) == 0:
                binary_acc.append(False)
                cls_acc.append(False)
                continue
            
            valid_total += 1
            pred = pred[0]
            
            # binary acc
            if gt.lower() == "normal":
                binary_acc.append(pred.lower() == gt.lower())
            else:
                binary_acc.append(pred.lower() != "normal")
            cls_acc.append(pred.lower() == gt.lower())

        binary_acc = np.array(binary_acc)
        cls_acc = np.array(cls_acc)
        binary_acc = np.mean(binary_acc)
        cls_acc = np.mean(cls_acc)

        # print the results
        print(f"Binary accuracy: {_round(binary_acc)}%")
        print(f"CLS accuracy: {_round(cls_acc)}%")

        # save the results to txt
        with open(self.save_path, "w") as f:
            f.write(f"Valid Predictions: {valid_total}\n") 
            f.write(f"Binary accuracy: {_round(binary_acc)}%\n")
            f.write(f"CLS accuracy: {_round(cls_acc)}%\n")
            
        print(f"Evaluation results saved to {self.save_path}\n", "-"*40, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--test_gt_path", type=str, required=True, help="Path to the ground truth CSV file")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the result CSV file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the evaluation results (default: pred_path with _eval suffix)")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.pred_path.replace(".csv", "_eval.txt")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    CLSEvaluator(args).evaluate()