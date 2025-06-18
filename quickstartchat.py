from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

chatbot = ChatBot("Chatu")

trainer = ListTrainer(chatbot)

trainer.train([
    "Hello",
    "How are you?",
    "How can I help you?",
    "Hi, can I help you?",
    "Sure, I'd like to book a flight to Iceland.",
    "Your flight has been booked."
])

response = chatbot.get_response("I would like to book a flight.")

print(response)
