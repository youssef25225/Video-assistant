import streamlit as st
import extract_text
import summarizer
import quiz

st.markdown("""
<h1 style='text-align: center;'>
    <span style='color: white;'>Video</span>
    <span style='color: red;'> Assistant</span>
</h1>
""", unsafe_allow_html=True)

if "text" not in st.session_state:
    st.session_state.text = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

link = st.text_input("Enter YouTube link")

t1, t2, t3 = st.tabs(["Transcript", "Summary", "Chat"])

with t1:
    if st.button("Get Transcript"):
        if not link.strip():
            st.warning("Please enter a valid YouTube link.")
        else:
            try:
                with st.spinner("Extracting transcript..."):
                    text = extract_text.transcribe(link)
                    st.session_state.text = text
                    st.session_state.retriever = quiz.load_qa(text)
                    st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.text:
        st.text_area("Transcript", st.session_state.text, height=400)

with t2:
    if st.button("Summarize Video"):
        if st.session_state.text is None:
            st.warning("Please get the transcript first.")
        else:
            try:
                with st.spinner("Summarizing..."):
                    summary = summarizer.summarize(st.session_state.text)
                    st.session_state.summary = summary
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.summary:
        data = st.session_state.summary
        st.subheader("Key Points")
        for point in data["points"]:
            st.write("•", point)
        st.subheader("Detailed Summary")
        st.write(data["detailed"])

with t3:
    if st.session_state.text is None:
        st.info("Please get the transcript first.")
    else:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])

        question = st.chat_input("Ask a question about the video")

        if question:
            try:
                with st.spinner("Generating answer..."):
                    answer = quiz.get_answer(question, st.session_state.retriever)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate answer: {e}")