import pandas as pd
import os
import re

def parse_questions_file(questions_file_path):
    raw_lines = read_text_file(questions_file_path).splitlines()

    questions = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue  

        m = re.match(r'^(\d+[\.\)]\s+)(.*)$', s)
        if m:
            questions.append(m.group(2).strip())
            continue

        if s.startswith("- "):
            questions.append(s[2:].strip())
            continue

        questions.append(s)

    return questions


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_col_values_from_file(question_directory, answer_directory, filename):

    questions_file_path = os.path.join(question_directory, filename + '_questions.txt')
    answers_file_path = os.path.join(answer_directory, filename + '_answers.txt')

    questions = parse_questions_file(questions_file_path)
    count = len(questions)
    print(f"{filename}: parsed {count} questions")

    content_of_answer_file = read_text_file(answers_file_path)

    answers_chunks = re.split(r'\n(?=\d+\.\n)', content_of_answer_file)
    answers = []
    for a in answers_chunks:
        a = a.strip()
        if not a:
            continue
        parts = a.split('.\n', 1)
        if len(parts) == 2:
            answers.append(parts[1].strip())
        else:
            answers.append(a.strip())

    if len(answers) != count:
        print(f"[WARN] In {filename}: {count} questions but {len(answers)} answers")

    website_data = pd.read_csv('G28.csv')
    filtered_df = website_data[website_data['name'] == filename]
    if filtered_df.empty:
        print(f"[WARN] No URL found for filename {filename} in G28.csv")
        url = None
    else:
        url = filtered_df['link'].values[0]

    data = []
    for idx in range(min(count, len(answers))):
        row = {
            "Question": questions[idx],
            "Answer": answers[idx],
            "File": filename,
            "URL": url,
        }
        data.append(row)

    return data



q_directory = 'questions'
a_directory = 'answers'

all_files = [
    file_name.removesuffix('_questions.txt')
    for file_name in os.listdir(q_directory)
    if file_name.endswith('_questions.txt')
]


columns = ['Question', 'Answer', 'File']
df = pd.DataFrame(columns=columns)

start = 0
for i, file in enumerate(all_files[start:]):
    print(i, end=' ')
    rows = read_col_values_from_file(q_directory, a_directory, file)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    print(f"Question & Answer pairs for {file} have been added to dataframe.")

print(df)

df.to_csv('SS_StructuredQA.csv', index=False)





