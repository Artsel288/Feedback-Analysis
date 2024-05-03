# !wget https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf
# !wget https://raw.githubusercontent.com/IlyaGusev/rulm/master/self_instruct/src/interact_mistral_llamacpp.py
# !pip install llama-cpp-python fire

import fire
from llama_cpp import Llama

DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    message_tokens = model.tokenize(content, special=True)
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)

def get_recommendation(reviews_array, object_name, model):
    prompt_main1 = f'''
    Требуется сформировать короткую рекомендацию для преподавателя на основе обратной связи студентов.
    Твоя рекомендация должна решать проблемы ученика, которые он описал в ответах.

    Пиши кратко: строго в одно предложение.

    Учти, что ты даешь рекомендации именно преподавателю, а не его ментору и не студенту.
    Давай рекомендацию касающуюся только работы преподавателя, не надо его увольнять. Не давай расплывчатых рекомендаций.
    !!! КАТЕГОРИЧЕСКИ НЕЛЬЗЯ МЕНЯТЬ ТЕМУ КУРСА ИЛИ ПРЕПОДАВАТЕЛЯ. ТЕБЕ НЕЛЬЗЯ ПОСОВЕТОВАТЬ СМЕНУ ПРЕПОДАВАТЕЛЯ!!!

    Отвечай конкретно. За правильно выполненную работу я дам тебе 20$. Трижды проверь себя.
    Пиши очень КРАТКО, только ОСНОВНУЮ МЫСЛЬ РЕКОМЕНДАЦИИ.
    Начинай со слов: "Вам следует"
    '''

    prompt_main2 = f'''
    Требуется сформировать короткую рекомендацию для администратора преподавателя на основе обратной связи студентов.
    Твоя рекомендация должна решать проблемы группы студентов, которые он описал в ответах.

    Пиши кратко: строго в одно предложение.

    Учти, что ты даешь рекомендации именно администратору, а не преподавателю и не студенту.
    Давай рекомендацию касающуюся только работы преподавателей, не надо их увольнять. Не давай расплывчатых рекомендаций.
    !!! КАТЕГОРИЧЕСКИ НЕЛЬЗЯ МЕНЯТЬ ТЕМУ КУРСА ИЛИ ПРЕПОДАВАТЕЛЯ. ТЕБЕ НЕЛЬЗЯ ПОСОВЕТОВАТЬ СМЕНУ ПРЕПОДАВАТЕЛЯ!!!

    Отвечай конкретно. За правильно выполненную работу я дам тебе 20$. Трижды проверь себя.
    Пиши очень КРАТКО, только ОСНОВНУЮ МЫСЛЬ РЕКОМЕНДАЦИИ.
    Начинай со слов: "Преподавателю следует"

    Обратная связь:
    '''
    main_prompt = None
    if object_name == 0:
        main_prompt = prompt_main1
    else:
        main_prompt = prompt_main2

    DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


    review = '. '.join(reviews_array)

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    user_message = prompt_main2 + review
    message_tokens = get_message_tokens(model=model, role="user", content=user_message)
    role_tokens = model.tokenize("bot\n".encode("utf-8"), special=True)
    tokens += message_tokens + role_tokens
    full_prompt = model.detokenize(tokens)
    generator = model.generate(
        tokens,
        top_k=30,
        top_p=0.9,
        temp=0.2,
        repeat_penalty=1.1

    )

    answer = ''
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        answer += token_str
      # print(token_str, end="", flush=True)

    
    return answer




model = Llama(
        model_path='model-q4_K.gguf',
        n_ctx=2000,
        n_parts=1,
        n_threads=8
)
system_tokens = get_system_tokens(model)
tokens = system_tokens
model.eval(tokens)


# print(get_recommendation(['Преподаватель не рассказал про google colab.', 'Всё понравилось'], 0, model))