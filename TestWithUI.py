import gradio as gr

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from bot import Bot
from question_answering_bot import manage_query
from set_key import set_api_key_from_file


def HCS(message, history):
    initial_msg = "You are talking to SFU counselling information Chatbot. If you wish to exit the conversation, please type in the word: exit"
    if len(history) == 0:
        print("Bot>> " + initial_msg, file=f)
        print(file=f)
        return initial_msg
    else:
        augmented_prompt = manage_query(message)
        ai_message = bot.handle_input(HumanMessage(content=augmented_prompt))

        print("User>>" + message, file=f)
        print(file=f)
        print("Bot>> " + ai_message.content, file=f)
        print(file=f)

        return ai_message.content



f = open(f"UIchatlog/chat1.txt", "a")
set_api_key_from_file()

# Initiate chatbot
bot_sysMessage = "You are an information chatbot to answer students' questions based on content that is given to you from SFU counselling website."
bot = Bot(bot_sysMessage)

gr.ChatInterface(
    HCS,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="title",
    description="description",
    theme="soft",
    #examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear and Exit",
).launch()