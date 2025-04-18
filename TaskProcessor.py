import json
import sys
import time
import random
import re
from math import ceil
from openai import OpenAI
import os
from collections import defaultdict
import base64

class TaskProcessor:
    def __init__(self, config_path="config.json", example_path = "WebVoyager_test.jsonl"):
        """Load configuration from the config file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("Configuration file not found!")
            sys.exit(1)
        try:
            with open(example_path, "r", encoding="utf-8") as f:
                self.example = [json.loads(line.strip()) for line in f if line.strip()]
        except FileNotFoundError:
            print("Example file not found!")
            sys.exit(1)
        self.client = OpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"]
        )

        self.model_name = self.config["model"]
        self.model_failure_name = self.config["model_failure"]
        self.temperature = self.config["temperature"]
        self.top_p = self.config["top_p"]
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.failure_trajectory = self.config["FAILUR_TRAJECTORY"]
        self.prompt_file_path = self.config["PROMPT_FILE_PATH"]
        self.save_path = self.config["SAVE_PATH"]
        self.prompts = self.load_prompts()

        # Load Task Configuration
        self.workflow_domains = self.config.get("workflow_domains", [])
        self.domain_categories = self.config.get("domain_categories", [])
        self.domain_task_configs = self.config.get("domain_task_configs", {})
        self.num_iterations = self.config.get("num_iterations")
        self.num_per_tasks = self.config.get("num_per_tasks")
        self.confidence_threshold = self.config.get("confidence_threshold")
        # Task Scaling Configuration
        self.task_scaling = self.config.get("task_scaling", {})
        self.low_ratio_start = self.task_scaling.get("low_ratio_start")
        self.low_ratio_end = self.task_scaling.get("low_ratio_end")
        self.medium_ratio_start = self.task_scaling.get("medium_ratio_start")
        self.medium_ratio_end = self.task_scaling.get("medium_ratio_end")
        self.high_ratio_start = self.task_scaling.get("high_ratio_start")
        self.high_ratio_end = self.task_scaling.get("high_ratio_end")

    def load_prompts(self):
        try:
            with open(self.prompt_file_path, "r", encoding="utf-8") as f:
                prompt_data = f.read().split("\n---\n")  # 以 `---` 分隔不同 Prompt
                prompts = {}

                for prompt in prompt_data:
                    if "Task Generation & Evaluation Workflow Prompt" in prompt:
                        prompts["Prompt 1"] = prompt.strip()
                    elif "Failure Analysis & Task Adjustment Workflow Prompt" in prompt:
                        prompts["Prompt 2"] = prompt.strip()
                    elif "Task Open-ended Judging Prompt" in prompt:
                        prompts["Prompt 3"] = prompt.strip()

                if "Prompt 1" not in prompts or "Prompt 2" not in prompts or "Prompt 3" not in prompts:
                    raise ValueError("Missing required prompts: Ensure `Prompt 1` and `Prompt 2` and `Prompt 3`exist.")

                return prompts
        except FileNotFoundError:
            print("Prompt file not found!")
            sys.exit(1)

    def call_gpt(self, prompt, max_retries=3,input_model=None):
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=input_model if input_model else self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )

                if response and hasattr(response, "choices") and response.choices:
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Call_GPT Error (Attempt {retries + 1}/{max_retries}): {e}")
                time.sleep(2)
                retries += 1
        print("Failed to get response from GPT after multiple attempts.")
        return None
    def call_gpt_with_images(self, text_prompt, image_paths, input_model=None):
        image_blocks = []
        for path in image_paths:
            image_obj = self.make_image_url_dict(path)
            if image_obj:
                image_blocks.append({
                    "type": "image_url",
                    "image_url": image_obj
                })
        if not image_blocks:
            print("[INFO] No valid images found. Falling back to text-only GPT call.")
            return self.call_gpt(text_prompt)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text_prompt}] + image_blocks
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=input_model or self.model_failure_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Vision GPT call failed: {e}")
            return None

    def save_tasks(self, tasks, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    existing_tasks = json.load(f)
                    if not isinstance(existing_tasks, list):
                        existing_tasks = []
                except json.JSONDecodeError:
                    existing_tasks = []
        else:
            existing_tasks = []

        all_tasks = existing_tasks + tasks

        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_tasks, f, indent=4, ensure_ascii=False)
        print(f"Tasks appended and saved to {path}")


    def get_difficulty_ratio(self, t, N):
        """计算每个网站生成任务的难度比例"""
        low_ratio = max(self.low_ratio_end,
                        self.low_ratio_start - (t / N) * (self.low_ratio_start - self.low_ratio_end))
        medium_ratio = min(self.medium_ratio_end,
                           self.medium_ratio_start + (t / N) * (self.medium_ratio_end - self.medium_ratio_start))
        high_ratio = min(self.high_ratio_end, (t / N) * self.high_ratio_end)
        return {"low": low_ratio, "medium": medium_ratio, "high": high_ratio}

    def group_examples_by_web(self):
        """将样例按 web_name 分组"""
        grouped = defaultdict(list)
        for ex in self.example:
            key = ex["web_name"]
            grouped[key].append(ex)
        return grouped


    def generate_task(self, t=0, domain=None, num_tasks=None, sample_ratio_per_web=None):
        if "Prompt 1" not in self.prompts:
            print("Error: 'Prompt 1' not found in loaded prompts.")
            return None

        N = self.num_iterations
        difficulty_ratio = self.get_difficulty_ratio(t, N)
        save_path = self.get_iteration_save_path(t,domain)

        grouped_examples = self.group_examples_by_web()
        all_generated_tasks = []

        for web_name, example_list in grouped_examples.items():
            print(f"\nProcessing Web: {web_name}, #examples: {len(example_list)}")

            start_url = example_list[0]["web"]  # 同一个网站 URL 是一致的
            if sample_ratio_per_web is not None and 0 < sample_ratio_per_web < 1.0:
                sample_size = max(1, int(len(example_list) * sample_ratio_per_web))
                example_list = random.sample(example_list, sample_size)
            total_num = num_tasks or len(example_list)
            task_count = min(total_num, len(example_list))  

            num_low = ceil(task_count * difficulty_ratio["low"])
            num_medium = ceil(task_count * difficulty_ratio["medium"])
            num_high = task_count - num_low - num_medium

            random.shuffle(example_list)

            difficulty_buckets = {
                "low": example_list[:num_low],
                "medium": example_list[num_low:num_low + num_medium],
                "high": example_list[num_low + num_medium:num_low + num_medium + num_high]
            }

            generated_tasks = []
            for difficulty, examples in difficulty_buckets.items():
                for task in examples:
                    task_template = {
                        "task": f"{task['ques']} on {web_name}",
                        "start_url": start_url,
                        "max_steps": "Integer",
                        "task_difficulty": difficulty,
                        "workflow_domain": random.sample(self.workflow_domains, k=2) if len(self.workflow_domains) >= 2 else self.workflow_domains,
                        "domain": domain or self.domain_categories,
                        "steps": {},
                        "final_action": "<submit/screenshot_only>",
                    }

                    prompt = self.prompts["Prompt 1"].format(
                        web=web_name,
                        num_tasks=1,
                        difficulty_ratio=json.dumps({difficulty: 1.0}, indent=4),
                        task_examples=json.dumps(task_template, indent=4)
                    )

                    gpt_response = self.call_gpt(prompt)
                    if gpt_response:
                        print(f"[GPT] {web_name}-{difficulty} response:")
                        print(gpt_response)
                        try:
                            response = gpt_response.strip().replace("```json", "").replace("```", "")
                            response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response)
                            if not response.startswith("[") and not response.endswith("]"):
                                response = "[" + response.replace("}{", "},{") + "]"
                            parsed_tasks = json.loads(response)
                            if isinstance(parsed_tasks, dict):
                                parsed_tasks = [parsed_tasks]
                            parsed_tasks, confidence = self.filter(parsed_tasks)
                            if confidence >= self.confidence_threshold:
                                continue
                            generated_tasks.extend(parsed_tasks)
                        except json.JSONDecodeError as e:
                            print(f"[ERROR] JSONDecodeError: {e} | Skip {web_name}-{difficulty}")
                    else:
                        print(f"[WARN] No GPT response for {web_name}-{difficulty}")
            
            self.save_tasks(generated_tasks, save_path)
            all_generated_tasks.extend(generated_tasks)

        return all_generated_tasks



    def select_failed_cases(self):
        try:
            with open(self.failure_trajectory, "r", encoding="utf-8") as f:
                failed_cases = json.load(f)

            failed_tasks = []
            failure_trajectories = []
            domain_failure_count = {} 

            for case in failed_cases:
                failed_tasks.append({
                    "task": case["task"],
                    "start_url": case["start_url"],
                    "max_steps": case["max_steps"],
                    "task_difficulty": case["task_difficulty"],
                    "workflow_domain": case["workflow_domain"],
                    "domain": case["domain"],
                    "final_action": case["final_action"],
                })

                failure_info = {
                    "task": case["task"],
                    "failure_reason": case["failure_trajectory"]["failure_reason"],
                    "failed_step": case["failure_trajectory"]["failed_step"],
                    "failure_path": case["failure_trajectory"]["failure_path"]
                }


                if "failure_screenshot_path" in case:
                    failure_info["failure_screenshot_path"] = case["failure_screenshot_path"]

                failure_trajectories.append(failure_info)

                # 统计失败领域
                domain = case["domain"]
                if domain not in domain_failure_count:
                    domain_failure_count[domain] = 0
                domain_failure_count[domain] += 1

            return failed_tasks, failure_trajectories, len(failure_trajectories), domain_failure_count

        except FileNotFoundError:
            print("Error: failure_trajectory.json not found！")
            return [], [], 0, {}
        except json.JSONDecodeError as e:  
            print(f"JSON Decode Error: {e}") 
            return [], [], 0, {}

        
    # filter fun
    def filter(self,parsed_task):
        if "Prompt 3" not in self.prompts:
            print("Error: 'Prompt 3' not found in loaded prompts.")
            return None
        for task in parsed_task:
            # critic by llm to judge if task is a opentoend task
            prompt = self.prompts["Prompt 3"].format(
                task_examples=json.dumps(task, indent=4)
            )
            gpt_response = self.call_gpt(prompt)
            print(gpt_response)
            # get confidence_score
            try:
                gpt_response = gpt_response.strip()
                gpt_response = gpt_response.replace("```json", "").replace("```", "")
                gpt_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', gpt_response)

                # 尝试解析为 JSON 对象
                parsed_response = json.loads(gpt_response)

                # 如果是 list，取第一个；如果是 dict，直接用
                if isinstance(parsed_response, list):
                    parsed_response = parsed_response[0]

                confidence = parsed_response.get("confidence_score", None)
                if confidence is not None:
                    print(f"[INFO] Task Confidence Score: {confidence}")
                else:
                    print("[WARNING] No confidence_score found in response.")
            except Exception as e:
                print(f"[ERROR] Failed to parse GPT response: {e}")
                print("[WARNING] No confidence_score found in response.")
                confidence = -1
            task["confidence_score"] = confidence  # 可以附加回 task 中
        return parsed_task,confidence

    def get_iteration_save_path(self, iteration: int,domain=None):
        """根据 iteration 数字生成带编号的保存路径，如 web_tasks_it0.json"""
        base_name = os.path.splitext(os.path.basename(self.save_path))[0]  # web_tasks
        extension = os.path.splitext(self.save_path)[1] or ".json"         # .json
        if domain:
            return f"{base_name}_{domain}{extension}"    
        return f"{base_name}_it{iteration}{extension}"                     # web_tasks_it{iteration}.json

    def adaptive_task_iteration(self,ratio=None):
        """Adaptive Task Iteration"""
        for t in range(1, self.num_iterations):
            print(f"\nAdaptive Iteration {t}/{self.num_iterations-1}")

            failed_tasks, failure_trajectories,total_num_failure,domain_failure_count = self.select_failed_cases()
            modified_tasks = []
            
            
            print(total_num_failure,domain_failure_count)
            # 根据失败轨迹生成修改轨迹
            for i, failed_task in enumerate(failed_tasks):
                failure_info = failure_trajectories[i]
                failure_screenshot_paths = failure_info.get("failure_screenshot_path", [])
                text_prompt = self.prompts["Prompt 2"].format(
                    failed_task=json.dumps(failed_task, indent=4),
                    failure_trajectory=json.dumps(failure_info, indent=4)
                )

                response = self.call_gpt_with_images(text_prompt, failure_screenshot_paths, input_model=self.model_failure_name)


                if response:
                    try:
                        # handle json format
                        response = response.strip().replace("```json", "").replace("```", "")
                        response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response)
                        if not response.startswith("[") and not response.endswith("]"):
                            response = "[" + response.replace("}{", "},{") + "]"
                        parsed_task = json.loads(response)
                        if isinstance(parsed_task, dict):
                            parsed_task = [parsed_task] 
                        parsed_task,confidence = self.filter(parsed_task)
                        if confidence >= self.confidence_threshold:
                            continue
                        modified_tasks.extend(parsed_task)
                        print(parsed_task)
                    except json.JSONDecodeError:
                        print(f"JSONDecodeError: Skipping modification for {failed_task['task']}")
                        continue
            it_save_path = self.get_iteration_save_path(t)
            self.save_tasks(modified_tasks,it_save_path)
            #  并生成新的任务
            self.generate_task(t,sample_ratio_per_web = ratio)

    def generate_tasks_by_domain_config(self, t=0):
        """Generate tasks in bulk based on domain_task_configs in config"""
        all_generated = []
        for domain, config in self.domain_task_configs.items():
            count = config.get("count", 3)
            print(f"\nGenerating tasks for domain: {domain}, count: {count}")
            tasks = self.generate_task(t=t, domain=domain, num_tasks=count)
            all_generated.extend(tasks)

        return all_generated
    def encode_image_to_base64(self, image_path):
        """将本地图像编码为 base64 字符串（失败时返回 None）"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {image_path}")
            return None
        
    def make_image_url_dict(self, image_path):
        """生成符合 GPT vision 的 image_url 数据结构"""
        b64 = self.encode_image_to_base64(image_path)
        if b64:
            return {"url": f"data:image/png;base64,{b64}"}
        return None
if __name__ == '__main__':
    # Instantiation, built-in parameters config path and example path,default is config.json, WebVoyager_test.jsonl
    task_processor = TaskProcessor()
    # Example Usage
    # t=0 for initial task generation, default for all cases in the example to generate the specified number of tasks
    # finally will generate web_task_it0.json file
    # gpt_response = task_processor.generate_task(t=0)


    # the task will be generated a specified number of times in a loop, the number of times can be adjusted in the config
    # you can add the ratio in the parameter, which can be used to randomly select a percentage of the number of tasks in the original example.
    # the final numbers are the ratio*num_example*num_per_tasks(works if u donnt wanna so much tasks in later loop)
    task_processor.adaptive_task_iteration(ratio=0.5)


    # Specified cases can be generated according to different domains, t can be used in the simulation of different loop stages, i.e., to control the difficulty
    # task_processor.generate_tasks_by_domain_config(t=2)
