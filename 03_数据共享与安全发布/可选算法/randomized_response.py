"""二元问题的随机响应算法。

算法背景：随机响应最初用于敏感问卷调查。参与者不总是直接报告真实答案，而是按已知概率
报告真实值，其余情况下报告一次随机答案。观察者难以从单条响应确定个人真实答案，但可以
根据扰动概率校正总体比例；未裁剪的校正估计无偏。

机制定义：设真实答案x属于{0,1}。以概率p返回x，以概率1-p返回一次公平抛硬币的结果。
若真实值为1的总体比例为π，则观测均值q满足q=pπ+(1-p)/2，因此可用
π=(q-(1-p)/2)/p进行估计。示例把有限样本产生的越界估计裁剪到[0,1]，
以保持比例取值合理，但裁剪后的估计通常有偏，不能继续称为无偏估计。

输入与输出：输入是一组二元真实答案、诚实回答概率p和随机种子；输出是一组扰动后的二元
回答。估计函数输入扰动结果和p，输出总体真值比例的估计。

作用与局限：p越小，单人的答案越难判断，但总体估计方差越大；p必须大于0且不超过1。
该基本机制只处理二元变量，依赖参与者正确执行随机化，并且不能保留记录之间的复杂关系。
小样本下估计会明显波动；重复收集同一个人的响应还可能削弱保护效果。
p=1会原样返回真值，没有扰动保护；本文件固定种子仅用于课堂复现，知道种子的观察者能
重现随机过程，因此不能把演示结果作为真实隐私发布。真实应用须使用安全随机源。
"""

import random

# 独立演示数据：1表示“是”，0表示“否”。
DEMO_TRUE_ANSWERS = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1] * 20
DEMO_TRUTH_PROBABILITY = 0.65
DEMO_RANDOM_SEED = 510


def randomized_response(
    true_answers: list[int],
    truth_probability: float,
    seed: int,
) -> list[int]:
    """按给定概率报告真值，否则报告一次公平随机的二元值。"""
    if not 0 < truth_probability <= 1:
        raise ValueError("truth_probability必须在(0, 1]范围内")
    if any(answer not in (0, 1) for answer in true_answers):
        raise ValueError("随机响应示例只接受0或1")

    generator = random.Random(seed)
    responses: list[int] = []
    for answer in true_answers:
        if generator.random() < truth_probability:
            responses.append(answer)
        else:
            responses.append(generator.randint(0, 1))
    return responses


def estimate_true_proportion(responses: list[int], truth_probability: float) -> float:
    """返回裁剪到[0,1]的比例估计；裁剪前的校正估计无偏，裁剪后通常有偏。"""
    if not 0 < truth_probability <= 1:
        raise ValueError("truth_probability必须在(0, 1]范围内")
    if any(answer not in (0, 1) for answer in responses):
        raise ValueError("responses只接受0或1")
    if not responses:
        raise ValueError("responses不能为空")
    observed_proportion = sum(responses) / len(responses)
    estimate = (observed_proportion - (1 - truth_probability) / 2) / truth_probability
    return min(1.0, max(0.0, estimate))


if __name__ == "__main__":
    demo_responses = randomized_response(
        DEMO_TRUE_ANSWERS, DEMO_TRUTH_PROBABILITY, DEMO_RANDOM_SEED
    )
    print("真实比例:", sum(DEMO_TRUE_ANSWERS) / len(DEMO_TRUE_ANSWERS))
    print("响应比例:", sum(demo_responses) / len(demo_responses))
    print("校正估计:", estimate_true_proportion(demo_responses, DEMO_TRUTH_PROBABILITY))
