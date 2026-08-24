from main import client, generate_answer

fake_top_chunks = [
    (0.9, 0, "Lionel Messi won the gold medal at the 2008 Summer Olympics with Argentina."),
    (0.8, 1, "Messi has won eight Ballons d'Or, more than any other player in history."),
]

question = "When did Messi win an Olympic medal?"

answer = generate_answer(client, question, fake_top_chunks)
print(answer)