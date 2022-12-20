import datasets
import pandas as pd
import streamlit as st


@st.cache
def load_dataset():
    return datasets.load_dataset("openai/webgpt_comparisons")["train"]


st.title("WebGPT Comparisons Dataset Explorer")
dataset = load_dataset()

st.header("Dataset Description")
st.write("The dataset was made available on the [HF Hub](https://huggingface.co/datasets/openai/webgpt_comparisons) on 2022/12/20.")

st.subheader("Official Description")
st.write(dataset.description)

st.subheader("Statistics")
st.markdown(f"The dataset contains a single 'train' split, and has {len(dataset):,d} examples. "
            f"The examples are dictionaries with the following keys: {', '.join(dataset.features.keys())}.")

st.header("Example")
index = st.slider("Data Index", 0, len(dataset) - 1)
example = dataset[index]

st.subheader("Question")
st.text_area("Full Text", example["question"]["full_text"], label_visibility="collapsed")
st.caption("Source: {}".format(example["question"]["dataset"]))

t1, t2 = st.columns(2)

scores = example["score_0"], example["score_1"]

with t1:
    st.subheader("Candidate 1{}".format(" (✓)" if scores[0] > scores[1] else ""))
    st.metric("User Rating", "{:.0f}%".format((scores[0] + 1) / 2 * 100),
              help="The raw score ranges from -1.0 to 1.0. The value displayed here is normalized to the range 0.0 to 1.0.")
    st.markdown("#### Answer ####")
    st.text_area("Answer", example["answer_0"], label_visibility="collapsed", height=300)
    st.markdown(f"#### Quotes (Total: {len(example['quotes_0']['title'])}) ####")
    for title, extract in zip(example["quotes_0"]["title"], example["quotes_0"]["extract"]):
        with st.expander(title):
            st.write(extract)

with t2:
    st.subheader("Candidate 2{}".format(" (✓)" if scores[1] > scores[0] else ""))
    st.metric("User Rating", "{:.0f}%".format((scores[1] + 1) / 2 * 100),
              help="The raw score ranges from -1.0 to 1.0. The value displayed here is normalized to the range 0.0 to 1.0.")
    st.markdown("#### Answer ####")
    st.text_area("Answer", example["answer_1"], label_visibility="collapsed", height=300)
    st.markdown(f"#### Quotes (Total: {len(example['quotes_1']['title'])}) ####")
    for title, extract in zip(example["quotes_1"]["title"], example["quotes_1"]["extract"]):
        with st.expander(title):
            st.write(extract)