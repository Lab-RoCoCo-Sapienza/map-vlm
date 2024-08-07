class Table:

    def get_info_env(self,file_name):
        env_info = ""
        with open(file_name , 'r') as file:
            data = file.readlines()
            for line in data:
                env_info += line 
            return env_info
