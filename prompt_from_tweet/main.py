from prompt_from_tweet.agents import TweetToImageCrew

def generate_image_prompt_from_tweet(tweet_text: str) -> str:
    pipeline = TweetToImageCrew()
    prompt = pipeline.run(tweet_text)
    return prompt

if __name__ == "__main__":
    tweet_input = input("enter the tweet you want to turn into an image prompt: ")
    generated_prompt = generate_image_prompt_from_tweet(tweet_input)
    print("Tweet :", tweet_input)
    print("Prompt généré :", generated_prompt)
