#!/usr/bin/env python3
# source: https://github.com/unconv/ai-podcaster

import subprocess
import datetime
import random
import openai
import time
import json
import sys
import os
from elevenlabs import generate, set_api_key, save, RateLimitError
from langchain.schema import SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

if elevenlabs_key:
    set_api_key(elevenlabs_key)

print("## AI-Podcaster by Unconventional Coding ##\n")


if not os.path.exists("dialogs"):
    os.mkdir("dialogs")

if not os.path.exists("podcasts"):
    os.mkdir("podcasts")

voice_names = {
    "male": [
        "Adam",
        "Antoni",
        "Arnold",
        "Callum",
        "Charlie",
        "Clyde",
        "Daniel",
        "Ethan",
    ],
    "female": [
        "Bella",
        "Charlotte",
        "Domi",
        "Dorothy",
        "Elli",
        "Emily",
        "Gigi",
        "Grace",
    ],
}

voices = {}


def get_voice(name, gender):
    if name not in voices:
        voices[name] = random.choice(voice_names[gender])
        voice_names[gender].remove(voices[name])
    return voices[name]


system_prompt = """
You are about to create a podcast script discussing the insights derived from a research paper provided by the user. Your goal is to generate a conversational podcast script between two presenters—Adam and Ethan—based on the content of the user-provided research paper. The podcast aims to deliver engaging content while maintaining a professional and informative tone.

Objective: Discuss the key findings and implications from the ruser-provided research paper. The script should provide an overview of the paper's significance and its impact on the field.
Tone: Maintain a conversational yet authoritative tone. Adam and Ethan should engage the audience by discussing the paper's content with enthusiasm and expertise.

Key Sections to Cover:

    Introduction (Adam):
        Welcoming the audience and introducing the research paper.
        Providing context on the significance of the paper's topic.
    Summary of Research Paper (Ethan):
        Briefly summarizing the key points and main findings from the paper.
        Explaining the methodology used in the research.
    Analysis and Discussion (Adam and Ethan):
        Delving deeper into the implications of the research findings.
        Exchanging thoughts, opinions, and potential applications arising from the paper.
    Conclusion (Adam and Ethan):
        Summarize the key takeaways from the research paper.
        Discuss potential future implications, applications, or areas for further research based on the paper's findings.
    Audience Engagement (Adam and Ethan):
        Encouraging listeners to explore the paper for further details.
        Also, engage the user by encouraging their participation in the podcast discussion.
        Opening the floor for questions or comments from the audience.

Additional Notes:
    Use a blend of technical language and layman terms to make the content accessible to a wide audience.
    Keep the discussion engaging and avoid jargon overload.
    Ensure that each section flows naturally into the next, maintaining a coherent narrative throughout the script.
    Also ensure that the podcast conclude in exactly {number_of_dialogs} dialogues.

Important: Please use the retrieved content from the research paper to generate the dialogues between Adam and Ethan. Provide informative discussions while capturing the essence of the paper's content in a conversational manner.

Answer the users question as best as possible.

{format_instructions}
"""

number_of_dialogs = 20


def generate_dialog(paper_summaries, podcast_id, openai_api_key):
    summaries = [
        {
            "title": summary["title"],
            "summary": summary["abstract"] + "\n\n" + summary["summary"],
        }
        for summary in paper_summaries
    ]
    transcript_file_name = f"podcasts/podcast_{podcast_id}.txt"
    transcript_file = open(transcript_file_name, "w")

    dialogs = []

    response_schemas = [
        ResponseSchema(name="speaker", description="The name of the speaker."),
        ResponseSchema(
            name="gender", description="The gender of the speaker (male or female)."
        ),
        ResponseSchema(name="content", description="The content of the speech."),
    ]

    messages = [
        SystemMessagePromptTemplate(system_prompt),
        HumanMessagePromptTemplate.from_template("{summaries}"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    for _ in range(0, number_of_dialogs):
        prompt = ChatPromptTemplate(
            messages=messages,
            input_variables=["summaries"],
            partial_variables={
                "format_instructions": format_instructions,
                "number_of_dialogs": number_of_dialogs,
            },
        )

        chat_model = ChatOpenAI(
            temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key
        )

        _input = prompt.format_prompt(summaries=summaries)
        output = chat_model(_input.to_messages())

        messages.append(output)

        parsed_output = output_parser.parse(output.content)

        transcript_file.write(
            parsed_output["speaker"] + ": " + parsed_output["content"] + "\n"
        )

        dialogs.append(parsed_output)

    transcript_file.close()
    return (dialogs, transcript_file_name)


def generate_audio(speaker, gender, content, filename):
    audio = generate(
        text=content,
        voice=get_voice(speaker, gender.lower()),
        model="eleven_monolingual_v1",
    )
    save(audio, filename)  # type: ignore


def generate_podcast(
    paper_summaries, podcast_id=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
):
    dialog_files = []
    concat_file = open("concat.txt", "w")

    print("Generating transcript")

    dialogs, transcript_file_name = generate_dialog(paper_summaries, podcast_id)

    print("Generating audio")
    try:
        for i, dialog in enumerate(dialogs):
            filename = f"dialogs/dialog{i}.wav"

            generate_audio(
                dialog["speaker"], dialog["gender"], dialog["content"], filename
            )

            concat_file.write("file " + filename + "\n")
            dialog_files.append(filename)
    except RateLimitError:
        print("ERROR: ElevenLabs ratelimit exceeded!")

    concat_file.close()

    podcast_file_name = f"podcasts/podcast{podcast_id}.wav"

    print("Concatenating audio")
    subprocess.run(
        f"ffmpeg -f concat -safe 0 -i concat.txt -c copy {podcast_file_name}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    os.unlink("concat.txt")

    for file in dialog_files:
        os.unlink(file)
    return podcast_file_name, transcript_file_name


if __name__ == "__main__":
    with open("", "r") as file:
        paper_summaries = file.read()
    podcast_file_name, transcript_file_name = generate_podcast(paper_summaries)
    print("\n## Podcast is ready! ##")
    print("Audio: " + podcast_file_name)
    print("Transcript: " + transcript_file_name)
