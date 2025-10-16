from openai import OpenAI
import os
import time

class UserLLM:
    def __init__(self, **kwargs):
        self.llm = kwargs["llm"]
        self.url = kwargs["url"]
        self.api_key = kwargs["api_key"]
        self.model = kwargs.get("model", "gemini-2.5-pro")
        self.user_prompt = kwargs.get("user_prompt", "")

    def generate(self, history):
        user = OpenAI(
            api_key=self.api_key,
            base_url = self.url,
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
                    model = "gemini-2.5-pro",
                    messages = [
                        {"role": "system", "content": self.user_prompt},
                        *history,
                        {"role": "user", "content": "Now continue as assistant."}
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