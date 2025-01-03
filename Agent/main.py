import pprint
 
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
 
from courses import *
 
# pip install -r requirements.txt
 
# Step 1 (Optional): Add a custom tool
@register_tool('ai_course_query_courses')
class AICourseQueryCourses(BaseTool):
    # 这个工具的描述告诉代理它的功能。
    description = """查询选课系统当前可选的课程列表。所有的课程都会分为【必修】和【选修】。请注意，仅支持查询类别为【必修】或【选修】的课程。返回课程列表，包含课程ID、课程名称和课程类别."""
    # 这个工具的参数告诉代理它有什么输入参数。
    parameters = [{
        'name': 'category',
        'type': 'string',
        'description': '课程类别，可接受的值为【必修】或【选修】，如不提供则为全部课程。',
        'required': False
    }]
 
    def call(self, params: str, **kwargs) -> str:
        # 调用该工具时执行的方法
        print('SYSTEM: 正在调用 Agent `ai_course_query_courses`，参数:', params)
        # `params` 是由LLM代理生成的参数。
        category = json5.loads(params).get('category', None)
        if category:
            category = category.strip('"').strip()
        return json5.dumps({"courses": query_courses(category)}, ensure_ascii=False)
 
 
@register_tool('ai_course_query_selected_courses')
class AICourseQuerySelectedCourses(BaseTool):
    # 这个工具的描述告诉代理它的功能。
    description = """查询用户当前已选的课程列表。返回课程列表，包含课程ID、课程名称和课程类别。"""
 
    def call(self, params: str, **kwargs) -> str:
        # 调用该工具时执行的方法
        print('SYSTEM: 正在调用 Agent `ai_course_query_selected_courses`，参数:', params)
        # `params` 是由LLM代理生成的参数。
        return json5.dumps({"courses": query_selected_courses()}, ensure_ascii=False)
 
 
@register_tool('ai_course_select_course')
class AICourseSelectCourse(BaseTool):
    # 这个工具的描述告诉代理它的功能。
    description = """给定课程ID，为用户选定该课程。返回选课结果：成功或失败原因。
    """
    # 这个工具的参数告诉代理它有什么输入参数。
    parameters = [{
        'name': 'course_id',
        'type': 'int',
        'description': '课程ID',
        'required': True
    }]
 
    def call(self, params: str, **kwargs) -> str:
        # 调用该工具时执行的方法
        print('SYSTEM: 正在调用 Agent `ai_course_select_course`，参数:', params)
        # `params` 是由LLM代理生成的参数。
        course_id = int(json5.loads(params)['course_id'])
        return json5.dumps({"result": select_course(course_id)}, ensure_ascii=False)
 
 
@register_tool('ai_course_delete_course')
class AICourseDeleteCourse(BaseTool):
    # 这个工具的描述告诉代理它的功能。
    description = """给定课程ID，为用户删除（退选）所选课程。返回删除情况：成功或失败原因。"""
    # 这个工具的参数告诉代理它有什么输入参数。
    parameters = [{
        'name': 'course_id',
        'type': 'int',
        'description': '课程ID',
        'required': True
    }]
 
    def call(self, params: str, **kwargs) -> str:
        # 调用该工具时执行的方法
        print('SYSTEM: 正在调用 Agent `ai_course_delete_course`，参数:', params)
        # `params` 是由LLM代理生成的参数。
        course_id = int(json5.loads(params)['course_id'])
        return json5.dumps({"result": delete_course(course_id)}, ensure_ascii=False)
 
 
# Step 2: Configure the LLM you are using.
llm_cfg = {
    'model': 'Qwen2.5-14B',
    'model_server': 'http://10.58.0.2:8000/v1',
    'api_key': 'None',
  
}
 
 
system_instruction = '''你是一个AI智能选课助手，能够帮助学生用户使用自然语言完成学生的选课操作。你需要做的是理解用户的自然语言请求，调用工具完成以下操作：
1. 查询：带有筛选的查询，可以筛选必修或选修。你需要先调用工具 `ai_course_query_courses` 查询可用的课程列表，并根据现有课程和用户需求匹配合适的课程。
   1.1. 查询增强，根据描述返回用户最为感兴趣的课程。
   1.2. 选择增强：用户在选课和删除时提供的课程不准确时，智能提供可能用户想提的课程。
2. 选课：调用工具 `ai_course_select_course` 选择需要的课程，智能返回结果。选课建立在用户查询的基础上，如果用户直接提出选课需求，您需要先进行第1步的查询操作，为用户匹配最合适的课程。
   2.1. 成功返回选课结果。
   2.2. 未成功返回错误。
3. 删除（退课）：调用工具 `ai_course_delete_course` 删除选择的课程，智能返回结果。
   3.1. 成功返回删除课程的结果。
   3.2. 未成功返回错误。
4. 课表查看：调用工具 `ai_course_delete_course` 查询用户已经选择的课程，智能返回用户课表。
   4.1. 未选择课程列表：你还可以调用工具 `ai_course_query_courses` 查询可用的课程列表，并与用户已经选择的课程比较，列出用户还没有选择的课程。如果用户还有未选择的【必修】课程，请重点通知用户。
   4.2. 课程推荐：根据用户选择的课程以及用户还没有选择的课程，推荐1~2门用户可能感兴趣的课程。
 
注意：
* 选课和退课操作需要提供课程ID，你需要首先根据获取到的课程列表查询与用户提供的课程信息最匹配的课程ID，再执行选课或退课操作。
* 对于不存在课程列表中的课程，请不要编造或猜测对应的课程名称和课程ID，你需要做的是忠实于查询到的课程列表，为用户推荐最适合他的课程。
* 每次调用只能处理一门课，对于用户的批量选课退课操作，请多次调用相应工具完成这些请求，不要让用户频繁输入课程信息。请一定要确保用户提供的所有的课程都选定了，建议完成选择后查询一次课表，并检查是否用户要求的所有的课程都选上了。
* 曾经请求过的数据可能并不是最新的。对于用户的每一次新的请求，您需要尽可能多地调用相关工具来获取最新数据，用以支持你的回答。
* 请调用提供的工具完成用户请求，并在操作后给出具体执行的动作。'''
tools = ['ai_course_query_courses', 'ai_course_query_selected_courses', 'ai_course_select_course', 'ai_course_delete_course']
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)
 
# Step 4: Run the agent as a chatbot.
messages = []  # 这个存储聊天记录。
print('Type "clear" to clear the content, and "exit" to close this session.')
print('''AI智能选课助手: 你好！我是一个AI智能选课助手，可以帮助你完成以下操作：
 
1. 查询课程：根据你的需求（如必修或选修），查询当前选课系统的课程列表。
2. 选课：根据课程ID选择课程，告诉你选课结果。
3. 退课：根据课程ID删除已选课程，显示退课结果。
4. 查看课表：查看你当前选的所有课程，或者推荐可能感兴趣的课程。
 
如果你需要查询课程、选课、退课或查看课表，请告诉我你的具体需求，我会帮你操作。''')
while True:
    # 例如，输入查询“绘制一只狗并旋转90度”。
    query = input('(user) > ')
    if query == 'clear':
        messages = []
        print('SYSTEM: 您的信息已重置，欢迎提问！')
        continue
    elif query == 'exit':
        print('SYSTEM: See you!')
        break
    # 将用户查询添加到聊天记录中。
    messages.append({'role': 'user', 'content': query})
    response = []
    for response in bot.run(messages=messages):
        if response[-1]['role'] == 'function':
            # pprint.pprint(response[-1], indent=2)
            print(f"{response[-1]['name']}: {response[-1]['content']}")
    if response[-1]['role'] == 'assistant':
        print(f"AI智能选课助手: {response[-1]['content']}")
    else:
        print("SYSTEM: 操作完成。")
    # 将机器人响应添加到聊天记录中。
    messages.extend(response)