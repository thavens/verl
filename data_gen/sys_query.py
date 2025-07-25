import datasets
import numpy as np


def gen_query_example(qa_dataset):
    # select 10 without replacement, 5 for sys query, 5 for user query
    qa_pairs = np.random.choice(qa_dataset, size=10, replace=False)
    # pick between 1 and 9 examples to put as system information
    num_sys = np.random.randint(1, 10)
    sys_pairs = qa_pairs[:num_sys]
    user_pairs = qa_pairs[num_sys:]
    sys_data_str = "\n".join([f"Q: {pair['question']}\nA: {pair['answer_1']}" for pair in sys_pairs])
    user_data_str = "\n".join([f"Q: {pair['question']}\nA: {pair['answer_1']}" for pair in user_pairs])

    selected_pair = np.random.randint(0, 10)
    prompt = f"Was it the system prompt or the user prompt that contains the answer to the following question?\nQ: {qa_pairs[selected_pair]['question']}"
    messages = [
        {"role": "system", "content": f"{sys_data_str}"},
        {"role": "user", "content": f"{user_data_str}\n\n{prompt}"},
    ]

    answer = "the system" if selected_pair < num_sys else "the user"
    return {"messages": messages, "answer": answer}

def sys_query_encapsulate(query_example):
    messages = query_example["messages"]
    answer = query_example["answer"]
    return {
        "data_source": "sys_query",
        "prompt": messages,
        "ability": "Security",
        "reward_model": {"ground_truth": answer},
    }

def get_dataset():
    qa_dataset = datasets.load_dataset("thavens/qa-data", split="train").to_list()
    sys_query_data = []
    for i in range(3840):
        query_example = gen_query_example(qa_dataset)
        sys_query_data.append(sys_query_encapsulate(query_example))
    return datasets.Dataset.from_list(sys_query_data)