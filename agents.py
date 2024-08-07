from openai import OpenAI
from table import Table
client = OpenAI()


class Agents:
    def __init__(self,image,task_description):
        self.encoded_image = image
        self.task_description = task_description
 
    def single_agent_table_planning(self,model_type,file_name_table):
        env_info = Table().get_info_env(file_name_table)
        agent = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": "You are an assistent which is able accurately\
              describe the planning step to reach the required goal. You know how it is made the \
             enviroment from the information of the following table that have for each row a \
             representation of an object with the position in the following shape:\n\
                "+ env_info + "\n\
                Your answer will be as that in the following example adding the navigation operation \
                    (Turn , move ,walk) and containing only the atomic step with the object position \
                 and nothing else.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using \
                    the objects perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n"},
            {"role": "user", "content": "The goal is " + self.task_description},
        ],
        temperature=0,
        max_tokens=400,
        )
        return agent.choices[0].message.content

    def multi_agent_table_planning(self,model_type,file_name_table):
        env_info = Table().get_info_env(file_name_table)


        def enviroment_agent():

            agent = client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "You are an useful assistant that have the task \
                 to describe the essential objects present \
                in the enviroment described in the following table that have for each row a representation\
                 of an object with the position in the following shape:\n\
                 "+ env_info + "\n\
                    The response will be a list of items necessary and essential to perform the objective.\n\
                    For example based on the goal your answer will be: \n\
                    - object1  \n\
                    - object2 \n\
                    - ....\n\
                    - objectN  "
                    },
                {"role": "user", "content": "The goal is " + self.task_description},
            ],
            temperature=0,
            max_tokens=400,
            )
            return agent.choices[0].message.content

        enviroment_info =  enviroment_agent()
        
        agent = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": "You are an assistent which is able accurately describe the \
             planning step to reach the required goal.\n\
            You know how are the object that you can use and where are from the following information : " + enviroment_info + "\
             You will do a planning to execute the goal using only the information written.\n\
                Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                and containing only the atomic step with the position of the object and nothing else.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using the objects \
                    perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n"},
            {"role": "user", "content": "The goal is " + self.task_description},
        ],
        temperature=0,
        max_tokens=400,
        )
        return enviroment_info, agent.choices[0].message.content

    def single_agent_vision_planning(self):
        agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an assistent which is able accurately describe the navigation planning step\
                         to reach the required goal: " + self.task_description +".\n\
                    You know how are the object that you can use and where are from the image.\n\
                    You will do a planning to execute the goal using only the information obtained from the image.\n\
                        Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                              and containing only the atomic step with the position of the object and nothing else.\n\
                          Your answer will be a list of only steps that help a agent to reach the goal.\
                             Try to do precise information for each step but in atomic way\n\
                    For the navigation write also the position of the object where is the object and \
                        the operation to reach the object. for example 'walk to object that is position' "},
                            {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
            temperature = 0,
        )
        response = (agent.choices[0].message.content)
        return response
    
    def multi_agent_vision_planning(self):


        def enviroment_agent():
            print("__________-")
            agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                        "text":"You are an assistent which is able accurately describe the content of an image. \n\
                        In particular, you are able to capture the main objects present.\n\
                        Explore the image accurately as an expert and find all the object that you can see.\n\
                        in the image and provide the relations that exist between them. \n\
                        These relations are described in the form of a triple (subject, relation, object) \
                        and when you answer you are only expected to answer with triples and nothing else. \n\
                        When writing the triples, try to execute this task: " +self.task_description + "\n\
                        and verify the elements that you neeed to solve and write the relation of the objects in the image.\n\
                        For example, if in a scene there is a door, a table in front of the door and a book on the table \
                        with a pen right to it, your answer should be: \
                        1) (table, in front of, door) \n\
                        2) (book, on, table) \n\
                        3) (pen, on, table) \n\
                        4) (pen, right to, book) \n\
                        5) (book, left to, pen). \n\
                        At the end of the task, you must write a instruction to solve the task, in a way that you can\
                        help who read your answer to understand how to solve the task without knowing the scene.",},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            temperature=0,
            )

            response = (agent.choices[0].message.content)
            print(response)
            return response
        
        
        def sim_ground_agent():
            print("__________-")
            agent = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                        "text":"You are an assistent which is able accurately describe the content of an image. \n\
                            In particular, you are able to describe accurately the content of the image to make one understand \
                            all the details of the image without seeing it. \n\
                            You should describe how the scene it is made with high level description and precise instruction to solve\
                            the following task : " + self.task_description+"\n\
                            If the task contains ambiguity in the solution of the task , for example same objects of the same type,\
                            specify the position of the object in the image or in relation at other objects.\n"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encoded_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
            temperature=0,
            )

            response = (agent.choices[0].message.content)
            print(response)
            return response
        print("_______________\n")

        enviroment_info = enviroment_agent() + "\n" + sim_ground_agent()
        print(enviroment_info)

        
        agent = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an  helpful assistant which is able accurately describe the navigation planning step to reach the required goal.\n\
             You know how are the object that you can use and where are from the following information " + enviroment_info + "\
             You will do a planning to execute the goal using the information written.\n\
            Your answer will be a list of only steps that help a agent to reach the goal. Try to do precise information for each step but in atomic way\n\
            Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                and containing only the atomic step with the position of the object and nothing else.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using the objects \
                    perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n"},
            {"role": "user", "content": "The goal is " + self.task_description},
        ],
        temperature=0,
        max_tokens=300,
        )
        return enviroment_info , agent.choices[0].message.content