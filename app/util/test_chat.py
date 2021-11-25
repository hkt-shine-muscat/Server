from random import randint

response_chat_data = ["안녕하세요, 반가워요!", "반가워요.", "다시 만나요.", "고마워요."]


def test_response() -> str:
    idx = randint(0, 4)
    return response_chat_data[idx]
