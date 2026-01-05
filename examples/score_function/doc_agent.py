import re
import sys
import string
from typing import Union, List
from collections import Counter


def validate_search_action_format(text: str) -> tuple[bool, str]:
    pattern = re.compile(r"^<think>.*?</think>\s*<search>.*?</search>$", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct') if format_match else (False, 'format incorrect')

def validate_mm_fetch_action_format(text: str) -> tuple[bool, str, tuple[str, str]]:
    pattern = re.compile(r"^<think>.*?</think>\s*<fetch>(image|text),\s*([1-9]\d*)-th</fetch>$", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct', (format_match.group(1), format_match.group(2))) if format_match else (False, 'format incorrect', ("", ""))

def validate_sm_fetch_action_format(text: str) -> tuple[bool, str, tuple[str, str]]:
    pattern = re.compile(r"^<think>.*?</think>\s*<fetch>\s*([1-9]\d*)-th\s*</fetch>$", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct', ('image', format_match.group(1))) if format_match else (False, 'format incorrect', ("", ""))

def validate_answer_action_format(text: str) -> tuple[bool, str]:
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    format_match = re.fullmatch(pattern, text)
    return (True, 'format correct') if format_match else (False, 'format incorrect')


def check_refusal(action_val: str) -> bool:
    """
    检查模型输出的 answer 内容是否表达了“无解/拒答”的意图。

    Args:
        action_val: <answer> 标签内的原始字符串 (可以包含 LaTeX 格式)

    Returns:
        bool: True 表示是拒答，False 表示不是
    """
    if not action_val:
        return False

    # 1. 预处理：转小写
    text = action_val.lower()

    # 2. 清洗 LaTeX 和特殊符号
    # 这一步非常重要，因为模型可能输出 \[ \boxed{The problem is...} \]

    # 移除 LaTeX 命令 (如 \boxed, \text, \large 等，以 \ 开头的字母串)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # 移除 LaTeX 结构符号 (花括号、方括号、美元符、反斜杠)
    text = re.sub(r'[\{\}\[\]\$\\]', '', text)

    # 移除除了字母、数字、空格以外的所有字符 (标点符号等)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. 规范化空白字符 (将换行、多余空格合并为一个空格)
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. 核心意图匹配
    # 定义标准拒答短语 (根据你的 Prompt: "The problem is not answerable")
    target_phrase = "the problem is not answerable"

    # 使用 'in' 进行包含匹配，而不是 '=='
    # 这样可以容忍前缀干扰，例如 "the final answer is the problem is not answerable"
    if target_phrase in text:
        return True

    return False

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


def compute_score(solution_str, ground_truth, mm_fetch) -> dict:
    result_score = {'search': 0, 'fetch': 0, 'fetch_image': 0, 'fetch_text': 0, 'turn_response_length': len(solution_str.split())}

    response = solution_str
    if ground_truth is not None:
        is_unanswerable = (ground_truth == 'The problem is not answerable')
        valid_template, reason = validate_answer_action_format(response)
        if not valid_template:
            result_score.update({"overall": -1.0,
                                 "format": 0.0,
                                 "accuracy": 0.0,})
                                 # "reason": f'bad format: {reason}'
            return result_score

        answer_part = extract_answer(response)
        is_refusal = check_refusal(answer_part)
        if answer_part is not None:
            try:
                answer = remove_boxed(last_boxed_only_string(answer_part))
            except Exception as e:
                result_score.update({"overall": -1.0,
                        "format": 0.0,
                        "accuracy": 0.0,})
                        # "reason": f'find box error: {e}'
                return result_score
        else:
            result_score.update({"overall": -1.0,
                                 "format": 0.0,
                                 "accuracy": 0.0,})
                                 # "reason": f'cannot extract answer'
            return result_score

        f1_score = get_f1_score(answer, ground_truth)

        if is_unanswerable:
            if is_refusal:
                result_score.update({"overall": f1_score * 5,
                                     "format": 1.0,
                                     "accuracy": 0.0, })
            else:
                result_score.update({"overall": -2.0,
                                     "format": 1.0,
                                     "accuracy": 0.0, })
        else:
            if is_refusal:
                result_score.update({"overall": -5.0,
                                     "format": 1.0,
                                     "accuracy": 0.0, })
            else:
                result_score.update(
                    {"overall": f1_score * 5,
                     "format": 1.0,
                     "accuracy": f1_score, })
        return result_score
    else:
        valid_template, reason = validate_search_action_format(response)
        action = 'search'
        if not valid_template and mm_fetch:
            valid_template, reason, (modal, pid) = validate_mm_fetch_action_format(response)
            action = 'fetch' + '_' + modal
        if not valid_template and not mm_fetch:
            valid_template, reason, (modal, pid) = validate_sm_fetch_action_format(response)
            action = 'fetch' + '_' + modal
        if valid_template:
            result_score[action] += 1
            result_score.update({"overall": 0.0,
                                 "format": 1.0,
                                 })
        else:
            result_score.update({"overall": -1.0,
                                 "format": 0.0,
                                 })

        return result_score