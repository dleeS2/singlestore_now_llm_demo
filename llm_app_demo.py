import os
import time
from datetime import datetime
import ttkbootstrap as ttk
from PIL import Image, ImageTk
import openai
import whisper
import numpy as np
import singlestoredb as s2
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
import pyaudio
import keyboard  
import wave
from pydub import AudioSegment
from pydub.playback import play
import requests
from openai.embeddings_utils import get_embeddings
from elevenlabs import generate, stream
from elevenlabs import set_api_key
from apikeys import openai_apikey, elevenlabs_apikey
from s2_connection import user, password, host, port, database, embedding_table_name
from langchain.tools.sql_database.tool import *
os.environ["OPENAI_API_KEY"] = openai_apikey
openai.api_key = os.environ["OPENAI_API_KEY"]
set_api_key(elevenlabs_apikey)

embedding_model = 'text-embedding-ada-002'
gpt_model = 'gpt-3.5-turbo-16k'
whisper_model = whisper.load_model("base")
db_uri = f"singlestoredb://{user}:{password}@{host}:{port}/{database}"
s2_conn = s2.connect(db_uri)



# Notes:
# Initial tests show the conversation memory to be a bit stable.  
# Feel free to uncomment these fields to test (memory, suffix, input_variables, agent_executor_kwargs)


db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=2)
print(db)
# memory = ConversationBufferMemory(input_key='input', memory_key='history')
llm = ChatOpenAI(model_name= gpt_model, temperature = 0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# suffix = '''
#     Previous conversation history:
#     {history}
    
#     Instructions: {input}

#     (You do not need to use these peices of information if not relevant)
#     Always list the database tables first.  Then look at the relevant table schemas.
#     {agent_scratchpad}
#     '''

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    # input_variables = ['input', 'agent_scratchpad', 'history'],
    verbose=True,
    prefix= '''
    You are an agent designed to interact with a SQL database called SingleStore. This sometimes has Shard and Sort keys in the table schemas, which you can ignore. 
    \nGiven an input question, create a syntactically correct MySQL query to run, then look at the results of the query and return the answer. 
    \n If you are asked about similarity questions, you should use the DOT_PRODUCT function.
    
    \nHere are a few examples of how to use the DOT_PRODUCT function:
    \nExample 1:
    Q: how similar are the questions and answers?
    A: The query used to find this is:
    
        select question, answer, dot_product(question_embedding, answer_embedding) as similarity from embeddings;
        
    \nExample 2:
    Q: What are the most similar questions in the embeddings table, not including itself?
    A: The query used to find this answer is:
    
        SELECT q1.question as question1, q2.question as question2, DOT_PRODUCT(q1.question_embedding, q2.question_embedding) :> float as score
        FROM embeddings q1, embeddings q2 
        WHERE question1 != question2 
        ORDER BY score DESC LIMIT 5;
    
    \nExample 3:
    Q: In the embeddings table, which rows are from the chatbot?
    A: The query used to find this answer is:
    
        SELECT category, question, answer FROM embeddings
        WHERE category = 'chatbot';

    \nIf you are asked to describe the database, you should run the query SHOW TABLES        
    \nUnless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    \n The question embeddings, answer embeddings, and audio answer columns are very long, so do not show them unless specifically asked to.
    \nYou can order the results by a relevant column to return the most interesting examples in the database.
    \nNever query for all the columns from a specific table, only ask for the relevant columns given the question.
    \nYou have access to tools for interacting with the database.\nOnly use the below tools. 
    Only use the information returned by the below tools to construct your final answer.
    \nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again up to 3 times.
    \n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    \n\nIf the question does not seem related to the database, just return "I don\'t know" as the answer.\n,
    ''',
    # suffix= suffix, 
    
    format_instructions='''Use the following format:\n
    \nQuestion: the input question you must answer
    \nThought: you should always think about what to do
    \nAction: the action to take, should be one of [{tool_names}]
    \nAction Input: the input to the action
    \nObservation: the result of the action
    \n... (this Thought/Action/Action Input/Observation can repeat 5 times)
    \nThought: I now know the final answer
    \nFinal Answer: the final answer to the original input question
    \nSQL Query used to get the Answer: the final sql query used for the final answer'
    ''',
    top_k=3,
    max_iterations=10
    # agent_executor_kwargs = {'memory':memory}    
)

def record_audio(output_file, sample_rate=44100, chunk_size=4096, audio_format=pyaudio.paInt16, channels=1):
    audio = pyaudio.PyAudio()
    # Open the microphone stream
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    frames = []

    # Function to check if the space bar is pressed
    def is_space_pressed():
        return keyboard.is_pressed('space')

    # Record audio until the space bar is pressed
    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        # Check if the space bar is pressed and break the loop if it is
        if is_space_pressed():
            break

    print("Ended recording")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(audio_format))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()


class ChatGUI:
    def __init__(self, database):
        self.root = ttk.Window(themename="cyborg")
        self.root.title("Chat with your SingleStore Data")
        self.root.geometry("780x720")
        self.database = database

        # Load and display the image
        image = Image.open("s2logo.png")
        self.image_tk = ImageTk.PhotoImage(image)
        image_label = ttk.Label(self.root, image=self.image_tk)
        image_label.pack()

        # Labels and entry widgets
        ttk.Label(self.root, text="User Question:", font=("Arial", 22)).place(x=30, y=100)
        self.entry = ttk.Entry(self.root, font=("Arial", 22))
        self.entry.insert(0, f"Enter your database {self.database} question here")
        self.entry.pack(side=ttk.TOP, padx=30, pady=50, fill=ttk.X)

        ttk.Label(self.root, text="Chatbot Response:", font=("Arial", 22)).place(x=30, y=260)
        self.response_entry = ttk.Text(self.root, height=7, width=90, font=("Arial", 22))
        self.response_entry.pack(padx=30, pady=70, fill=ttk.X)

        ttk.Label(self.root, text="SQL Query used to get the Answer:", font=("Arial", 22)).place(x=30, y=550)
        self.sql_query_entry = ttk.Text(self.root, height=6, width=90, font=("Arial", 22))
        self.sql_query_entry.pack(side=ttk.BOTTOM, padx=30, pady=30, fill=ttk.X)

        # Buttons
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 22), background='SystemButtonFace')
        style.configure("TEntry", font=("Arial", 22), background='SystemButtonFace')

        ttk.Button(self.root, text="Mic", command=self.transcribe_mic, style="outline").place(x=45, y=200)
        ttk.Button(self.root, text="Chat", command=self.on_click, style="outline").place(x=340, y=200)
        ttk.Button(self.root, text="Reset", command=self.clear_text, style="outline").place(x=645, y=200)


    def transcribe_mic(self):
        # Record the start time
        total_time = time.time() 

        # Usage
        output_file = 'recording.wav'

        print('Started Recording. Press `Space Bar` to end recording')

        # Record the start time
        start_time = time.time()

        record_audio(output_file)
        print(f"Recording saved as {output_file}")

        # Calculate the elapsed time
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Execution time for audio recording: {elapsed_time:.2f} milliseconds")

        # Record the start time
        start_time = time.time()


        transcript = whisper_model.transcribe("recording.wav", fp16=False)

        # Calculate the elapsed time
        elapsed_time = (time.time() - start_time) * 1000
        print(f"Execution time for transcribing recording.wav file: {elapsed_time:.2f} milliseconds")

        self.entry.delete(0,ttk.END)
        self.entry.insert(0, transcript["text"])
        self.on_click()

        # Calculate the elapsed time
        total_elapsed_time = (time.time() - total_time) * 1000
        print(f"Total time: {total_elapsed_time:.2f} milliseconds")

    def on_click(self):
        # Get the query text from the entry widget
        query = self.entry.get()

        # Record the start time
        start_time = time.time()

        # Run the query using the agent executor
        result = insert_embedding(query)  # Call the method with self.
        
        print(result)
        # Find the position of "SQL Query used to get the Answer: "
        
        if result is not None:
            if isinstance(result, tuple):
                result = result[0]
            else:
                result = result
            
            query_start = result.find("SQL Query used to get the Answer: ")
            # Extract everything before the query
            description = result[:query_start]

            # Extract everything after the query
            sql_query = result[query_start + len("SQL Query used to get the Answer: "):]

            self.display_text(description)
            self.root.update()  # Update the GUI immediately
            self.display_text2(sql_query)
            self.root.update()  # Update the GUI immediately

            # Print the results
            print("Description:", description)
            print("SQL Query:", sql_query)

            # if using existing answer, I don't need to get new audio for description

            answer = description[:50]
            print(f'description:{answer}')

            params = {
                    'answer': answer,
                    }

            with s2_conn.cursor() as cur:
                # Record the start time
                start_time = time.time()
                stmt = f"""select audio_answer from embeddings where answer like %(answer)s limit 1"""
                # print(stmt)
                cur.execute(stmt, {'answer': '%' + params['answer'] + '%'})
                row = cur.fetchone()[0]
                # s2_conn.close()

                # Write the binary data to a file
                with open(f'output.wav', 'wb') as file:
                    file.write(row)
                
                elapsed_time = (time.time() - start_time) * 1000
                print(f"Execution time for bringing audio: {elapsed_time:.2f} milliseconds")

            # Calculate the elapsed time
            elapsed_time = (time.time() - start_time) * 1000
            print(f"Execution time for call back action: {elapsed_time:.2f} milliseconds")
            print("-"*50)

            sound = AudioSegment.from_file("output.wav")
            # play(sound)
            time.sleep(.2)
            # Schedule the audio streaming operation with a delay
            self.root.after(100, play(sound), description)  # Use self. to access the method

            return description, sql_query
        else:
            # Run the query using the agent executor
            description = result
            sql_query = 'No SQL query'

            self.display_text(description)
            self.root.update()  # Update the GUI immediately
            self.display_text2(sql_query)
            self.root.update()  # Update the GUI immediately

            print('else')
            answer = description[:50]
            print(f'description2:{answer}')

            params = {
                    'answer': answer,
                    }

            with s2_conn.cursor() as cur:
                # Record the start time
                start_time = time.time()
                stmt = f"""select audio_answer from embeddings where answer like %(answer)s limit 1"""
                print(stmt)
                cur.execute(stmt, {'answer': '%' + params['answer'] + '%'})
                row = cur.fetchone()[0]
                s2_conn.close()

                # Write the binary data to a file
                with open(f'output.wav', 'wb') as file:
                    file.write(row)
                
                elapsed_time = (time.time() - start_time) * 1000
                print(f"Execution time for bringing audio: {elapsed_time:.2f} milliseconds")


            # Calculate the elapsed time
            elapsed_time = (time.time() - start_time) * 1000
            print(f"Execution time for call back action: {elapsed_time:.2f} milliseconds")
            print("-"*50)

            sound = AudioSegment.from_file("output.wav")
            time.sleep(.2)
            # Schedule the audio streaming operation with a delay
            self.root.after(100, play(sound), result)  # Use self. to access the method

            return description, sql_query

    def clear_text(self):
        # Clear the entry widget
        self.entry.delete(0, ttk.END)
        self.entry.insert(0, f"Enter your quesion on database: {database}")

    def display_text(self, result):
        if result is not None:
            print("Inserting text:", result)  # Print the text you're trying to insert
            self.response_entry.delete("1.0", ttk.END)  # Clear existing text
            self.response_entry.insert("1.0", result)  # Insert the new text at the beginning
        else:
            print("Result is None, cannot insert.")

    def display_text2(self, result):
        if result is not None:
            print("Inserting text:", result)  # Print the text you're trying to insert
            self.sql_query_entry.delete("1.0", ttk.END)  # Clear existing text
            self.sql_query_entry.insert("1.0", result)  # Insert the new text at the beginning
        else:
            print("Result is None, cannot insert.")

    def run(self):
        self.root.mainloop()

def insert_embedding(question):
    print(f'\nQuestion asked: {question}')
    category = 'chatbot'
    
    # Record the start time
    start_time = time.time()

    question_embedding= [np.array(x, '<f4') for x in get_embeddings([question], engine=embedding_model)]

    # Calculate the elapsed time
    elapsed_time = (time.time() - start_time) * 1000
    print(f"Execution time for getting the question embedding: {elapsed_time:.2f} milliseconds")

    params = {
            'question_embedding': question_embedding,
            }
    # Check if embedding is similar to existing questions
    stmt = f'select question, answer, dot_product( %(question_embedding)s, question_embedding) :> float as score from embeddings where category="chatbot" order by score desc limit 1;'

    with s2_conn.cursor() as cur:
        # Record the start time
        start_time = time.time()
        
        cur.execute(stmt, params)
        row = cur.fetchone()
        # if no row it will error for now

        elapsed_time = (time.time() - start_time) * 1000
        print(f"Execution time for checking existing questions: {elapsed_time:.2f} milliseconds")

        try:

            question2, answer, score = row
            print(f"\nClosest Matching row:\nQuestion: {question2}\nAnswer: {answer}\nSimilarity Score: {score}")
    
            if score >.95:
                print('Action to take: Using existing answer')
                existing_answer = 1
                return answer, existing_answer

            else:
                print('Action to take: Running agent_executor')


                # Record the start time
                start_time = time.time()
                
                answer2 = agent_executor.run(question)

                # break out answer2
                if answer2 is not None:
                    query_start = answer2.find("SQL Query used to get the Answer: ")

                    # Extract everything before the query
                    description = answer2[:query_start]

                    # Extract everything after the query
                    sql_query = answer2[query_start + len("SQL Query used to get the Answer: "):]

                else:
                    # Run the query using the agent executor
                    description = answer2
                    sql_query = 'No SQL query'

                # Calculate the elapsed time
                elapsed_time = (time.time() - start_time) * 1000
                print(f"agent_executor execution time: {elapsed_time:.2f} milliseconds")

                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Record the start time
                start_time = time.time()

                answer_embedding = [np.array(x, '<f4') for x in get_embeddings([answer2], engine=embedding_model)]

                get_audio_answer(description)

                # Read the audio file as binary data
                with open('output.wav', 'rb') as file:
                    audio_answer = file.read()
                    
                # Calculate the elapsed time
                elapsed_time = (time.time() - start_time) * 1000
                print(f"Answer embeddings execution time: {elapsed_time:.2f} milliseconds")

                params = {'category': category, 'question': question,
                        'question_embedding': question_embedding,
                        'answer': answer2, 'answer_embedding': answer_embedding,
                        'audio_answer': audio_answer,
                        'created_at': created_at}
                
                # Send to SingleStoreDB
                stmt = f"INSERT INTO {embedding_table_name} (category, question, question_embedding, answer, answer_embedding, audio_answer, created_at) VALUES (%(category)s, \n%(question)s, \n%(question_embedding)s, \n%(answer)s, \n%(answer_embedding)s, \n%(audio_answer)s, \n%(created_at)s)"

                # Record the start time
                start_time = time.time()

                with s2_conn.cursor() as cur:
                    cur.execute(stmt, params)

                # Calculate the elapsed time
                elapsed_time = (time.time() - start_time) * 1000
                print(f"Insert to SingleStore execution time: {elapsed_time:.2f} milliseconds")
                existing_answer = 0

                # sound = AudioSegment.from_file("output.wav")
                # time.sleep(1)
                # play(sound)

                return answer2, existing_answer, description, sql_query
            
        except Exception as e:
            print(e)

def start_audio_stream(result):
    audio_stream = generate(
        text=result,
        voice="Hannah",
        model="eleven_monolingual_v1",
        stream=True
    )
    stream(audio_stream)

def get_audio_answer(description_answer):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/rrbP5uKy2DDmBwcTZmtV"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": elevenlabs_apikey
    }

    data = {
    "text": description_answer,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    with open('output.wav', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    # Add a delay to ensure the file is fully saved before playing
    time.sleep(1)  # You can adjust the delay time as needed

if __name__ == "__main__":
    database_name = "stock_pull"
    chat_app = ChatGUI(database_name)
    embedding_model = 'text-embedding-ada-002'
    chat_app.run()
