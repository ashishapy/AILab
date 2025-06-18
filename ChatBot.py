from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

chatbot = ChatBot("Chatu",
                  preprocessors=['chatterbot.preprocessors.clean_whitespace'],
                  logic_adapters=[
                      'chatterbot.logic.MathematicalEvaluation',
                      'chatterbot.logic.BestMatch',
                      'chatterbot.logic.TimeLogicAdapter'
                  ]
        )

trainer = ListTrainer(chatbot)

trainer.train([
    "Hello",
    "How are you?",
    "How can I help you?",
    "Hi, can I help you?"
])


response = chatbot.get_response("Good Morning!")

print(response)
