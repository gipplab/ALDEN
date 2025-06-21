import re
import sys
import string
from typing import Union, List
from collections import Counter


def validate_search_action_format(text: str) -> tuple[bool, str]:
    pattern = re.compile(r"<think>.*?</think>\s*<search>.*?</search>", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct') if format_match else (False, 'format incorrect')

def validate_fetch_action_format(text: str) -> tuple[bool, str]:
    pattern = re.compile(r"<think>.*?</think>\s*<fetch>.*?</fetch>", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct') if format_match else (False, 'format incorrect')

def validate_answer_action_format(text: str) -> tuple[bool, str]:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct') if format_match else (False, 'format incorrect')

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']


def compute_score(solution_str, ground_truth) -> dict:
    # handling both the base model and the instruction-tuned model
    # if "<|im_start|>assistant\n" in solution_str:
    #     solution_str_split = solution_str.split("<|im_start|>assistant\n")
    # else:
    #     solution_str_split = solution_str.split("Assistant:")
    # einfo, page_ids = extra_info
    # gt_page_ids = set(einfo['answer_page_idx'])
    # page_ids = set(page_ids)
    # hit_p = len(gt_page_ids.intersection(page_ids)) / (len(page_ids) + 1e-6)
    # hit_r = len(gt_page_ids.intersection(page_ids)) / (len(gt_page_ids) + 1e-6)
    # hit_f1 = 2 * hit_p * hit_r / (hit_p + hit_r + 1e-6)
    # result_score = {
    #     "hit_precision": hit_p,
    #     "hit_recall": hit_r,
    #     "hit_f1": hit_f1
    # }
    result_score = {'search': 0, 'fetch': 0, 'turn_response_length': len(solution_str.split())}

    response = solution_str
    if ground_truth is not None:
        valid_template, reason = validate_answer_action_format(response)
        if not valid_template:
            result_score.update({"overall": 0.0,
                                 "format": 0.0,
                                 "accuracy": 0.0,})
                                 # "reason": f'bad format: {reason}'
            return result_score

        # if response.endswith(tokenizer.eos_token):
        #     response = response[:-len(tokenizer.eos_token)]
        # else:
        #     return {"overall": 0.0,
        #             "format": 0.0,
        #             "accuracy": 0.0,
        #             "reason": f'over length'}

        answer_part = extract_answer(response)
        if answer_part is not None:
            try:
                answer = remove_boxed(last_boxed_only_string(answer_part))
            except Exception as e:
                result_score.update({"overall": 0.2,
                        "format": 0.2,
                        "accuracy": 0,})
                        # "reason": f'find box error: {e}'
                return result_score
        else:
            answer = ''
            result_score.update({"overall": 0.1,
                                 "format": 0.1,
                                 "accuracy": 0.0,})
                                 # "reason": f'cannot extract answer'
            return result_score

        f1_score = get_f1_score(answer, ground_truth)
        if f1_score > 0:
            result_score.update(
                {"overall": f1_score * 5 + 0.4,
                                 "format": 1.0,
                                 "accuracy": f1_score,})
                                 # "reason": f'correct answer, get f1 score: {f1_score}'
            return result_score
        else:
            result_score.update({"overall": 0.4,
                                 "format": 1.0,
                                 "accuracy": 0.0,})
                                 # "reason": f'wrong answer but good format: {answer}'
            return result_score
    else:
        valid_template, reason = validate_search_action_format(response)
        action = 'search'
        if not valid_template:
            valid_template, reason = validate_fetch_action_format(response)
            action = 'fetch'
        if valid_template:
            result_score[action] += 1
            result_score.update({"overall": 0.4,
                                 "format": 1.0,
                                 })
        else:
            result_score.update({"overall": 0.0,
                                 "format": 0.0,
                                 })

        return result_score
