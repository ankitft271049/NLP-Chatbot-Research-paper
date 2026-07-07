import aiml
bot=aiml.Kernel()
bot.learn("hi.aiml")
bot.learn("intents.aiml")

while True:
	print("ALICE: "+bot.respond(input("User> ")))
