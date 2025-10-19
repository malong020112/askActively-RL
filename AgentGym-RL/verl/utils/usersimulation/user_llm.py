from openai import OpenAI
import os
import time

class UserLLM:
    def __init__(self, **kwargs):
        self.user_model = kwargs["user_model"]
        self.api_url = kwargs["api_url"]
        self.api_key = kwargs["api_key"]
        self.user_item = kwargs.get("user_item", "")
        
        SYS_PROMPT_USER1 = """
        You will play the role of a user seeking help from an assistant. 
        Your goal is to have the assistant successfully solve the initial task you provide. 
        Below are the attributes and background of the problem you want help with:
        """

        SYS_PROMPT_USER2 = """
        Please follow these interaction rules carefully:

        1. You are acting purely as the user — not as the assistant.
        2. If the assistant determines your question is unclear, it will ask one clarifying question with several options.
        - Choose exactly **one** option that best matches your intended meaning and reply with its letter (e.g., “A”).
        3. If none of the provided options fit your situation, you may reply with:
        - “Either is fine.” (if multiple options could work), or a similar expression
        4. When the assistant gives a final answer, decide whether it correctly solves your original problem:
        - If you are satisfied, reply **exactly** with `"ACCEPT"`.
        - If you are not satisfied, reply **exactly** with `"REJECT"`.
        5. You should only respond to the assistant’s clarifying questions or its final answers — do not start new topics.
        6. Remember: your role is to represent the **user’s intent** faithfully and consistently throughout the conversation.
        """
        self.prompt = SYS_PROMPT_USER1 + f"\n{self.user_item}\n" + SYS_PROMPT_USER2

    def generate(self, history):
        user = OpenAI(
            api_key=self.api_key,
            base_url = self.api_url
        )

        reward = 0
        terminate = False
        user_response = ""

        max_retries = 5
        retries = 0
        delay = 1  # 初始延迟时间（秒）
        for attempt in range(max_retries):
            try:
                response = user.chat.completions.create(
                    model = self.user_model,
                    messages = [
                        {"role": "system", "content": self.prompt},
                        *history,
                        {"role": "user", "content": "Now continue as user."}
                    ],
                    stream = False
                )
                user_response = {
                    "role": "user",
                    "content": response.choices[0].message.content
                }
                break  # 成功则跳出重试循环
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"第 {attempt + 1} 次重试失败：{e}")
                    raise e  # 超过最大重试次数，抛出异常
                time.sleep(delay)  # 等待后重试
                delay *= 2  # 指数退避，每次等待时间翻倍
        
        if user_response["content"] == "ACCEPT":
            reward = 1
            terminate = True
        elif user_response["content"] == "REJECT":
            reward = -1
        else :
            reward = -0.3

        return (user_response, reward, terminate)
    
    def close(self):
        pass