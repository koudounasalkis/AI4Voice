import torch
from TTS.api import TTS
import glob
from tqdm import tqdm
import pandas as pd 
import random
import os 


DATASET = 'AVFAD'   # 'SVD', 'AVFAD' or 'IPV'

if __name__ == "__main__":
    
    ## Get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ## Load TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    ## Generate TTS samples
    if DATASET == 'SVD':
        df = pd.read_csv("data_svd/metadata.csv")
        df = df[df["Health"] == "healthy"]
        file_paths = df["File path"].values
        file_paths = [file_path.replace("\\", "/") for file_path in file_paths]

        files = sorted(glob.glob("data_svd/*.wav"))

        for file in tqdm(files):
            if file in file_paths:
                print(file)

                ## Get text
                if "phrase" in file:
                    text = "Guten Morgen, wie geht es Ihnen?"
                else:
                    text = "aaaaaaaaaaaaaaaaaaaaaaa"

                ## Get new file name
                new_file = file.replace("data_svd", "data_svd_tts")

                ## Synthesize
                if os.path.exists(new_file):
                    continue
                else:
                    tts.tts_to_file(
                        text=text, 
                        speaker_wav=file, 
                        language="de", 
                        file_path=new_file
                        )

    elif DATASET == 'AVFAD':
        df = pd.read_csv("data_avfad/metadata.csv")
        df = df[df["Healthy"] == False]
        file_paths = df["File Path"].values
        # random.shuffle(file_paths)
        # file_paths = file_paths[:260]
        file_paths = [file_path.replace("\\", "/") for file_path in file_paths]

        files = sorted(glob.glob("data_avfad/*.wav"))

        for file in tqdm(files):
            if file in file_paths:
                print(file)

                ## Get text
                if "sentence" in file:
                    text = "\
                        A Marta e o avô vivem naquele casarão rosa velho.\
                        Sofia saiu cedo da sala.\
                        A asa do avião andava avariada.\
                        Agora é hora de acabar.\
                        A minha mãe mandou-me embora.\
                        O Tiago comeu quatro peras.\
                        "
                else:
                    ## choose a random vowel to synthesize, between /a/, /e/, /u/
                    random_vowel = random.choice(["a", "e", "u", "aeu"])
                    if random_vowel == "a":
                        text = "aaaaaaaaaaaaaaaaaaaaaaa"
                    elif random_vowel == "e":
                        text = "eeeeeeeeeeeeeeeeeeeeeee"
                    elif random_vowel == "u":
                        text = "uuuuuuuuuuuuuuuuuuuuuuu"
                    else:
                        text = "\
                            aaaaaaaaaaaaaaaaaaaaaaa.\
                            eeeeeeeeeeeeeeeeeeeeeee.\
                            uuuuuuuuuuuuuuuuuuuuuuu.\
                            "

                ## Get new file name
                new_file = file.replace("data_avfad", "data_avfad_tts")

                ## Synthesize
                if os.path.exists(new_file):
                    continue
                else:
                    tts.tts_to_file(
                        text=text, 
                        speaker_wav=file, 
                        language="pt", 
                        file_path=new_file
                        )

    elif DATASET == 'IPV':
        files = sorted(glob.glob("data_ipv/*.wav"))
        for file in tqdm(files): 
            print(file)

            ## Get text
            if "cs" in file:
                text = "\
                    Il nuovo libro verde è sulla scatola.\
                    L'uomo e la donna mangiano le uova.\
                    Che cosa ha rotto il gatto?\
                    Le mie nonne non vanno mai al mare.\
                    Lo zoppo ha toccato il letto.\
                    "
            else:
                text = "aaaaaaaaaaaaaaaaaaaaaaa"
            
            ## Get new file name
            new_file = file.replace("data_ipv", "data_ipv_tts")

            ## Synthesize
            if os.path.exists(new_file):
                continue
            else:
                ## Synthesize
                tts.tts_to_file(
                    text=text, 
                    speaker_wav=file, 
                    language="it", 
                    file_path=new_file
                    )

    else:
        print("Invalid dataset")
        exit(1)