import logging
import os

from dotenv import load_dotenv

from openai_helper import OpenAIHelper
from telegram_bot import ChatGPT3TelegramBot


def main():
    load_dotenv()

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    openai_config = {
        'api_key': os.environ['OPENAI_API_KEY'],
        'show_usage': False,
        'max_history_size': 10,
        'max_conversation_age_minutes': 180,
        'assistant_prompt': 'You are a helpful assistant.',
        'max_tokens': 1200,
        'model': 'gpt-3.5-turbo',
        'temperature': 1,
        'n_choices': 1,
        'presence_penalty': 0,
        'frequency_penalty': 0,
    }

    telegram_config = {
        'token': os.environ['TELEGRAM_BOT_TOKEN'],
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
    }

    # Setup and run ChatGPT and Telegram bot
    openai_helper = OpenAIHelper(config=openai_config)
    telegram_bot = ChatGPT3TelegramBot(config=telegram_config, openai=openai_helper)
    telegram_bot.run()


if __name__ == '__main__':
    main()
