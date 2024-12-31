
# AI Cold Caller

The AI Cold Caller is an automated system designed to engage with customers over the phone, simulate sales conversations, and collect analytics based on customer interactions. It uses AI models for speech recognition, sentiment analysis, conversation generation, and text-to-speech synthesis.

## Features

- **Speech-to-Text**: Converts customer speech into text using the Whisper model.
- **Sentiment Analysis**: Analyzes customer sentiment during the conversation using sentiment-analysis models.
- **Response Generation**: Generates agent responses using a large language model (LLM).
- **Text-to-Speech**: Converts agent responses into speech and plays them to the customer using Google Text-to-Speech (gTTS).
- **Call Analytics**: Tracks various metrics like call duration, customer and agent talk time, sentiment scores, and positive/negative responses.
- **Conversation History**: Records the entire conversation between the agent and the customer.
- **Call Logging**: Saves conversation transcripts and analytics data for further analysis.

## Requirements

The AI Cold Caller requires the following Python libraries:

- `torch`
- `transformers`
- `whisper`
- `pyaudio`
- `wave`
- `numpy`
- `pandas`
- `gtts`
- `playsound`
- `datetime`

Install the dependencies using `pip`:

```bash
pip install torch transformers whisper pyaudio numpy pandas gtts playsound
```

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/ai-cold-caller.git
   cd ai-cold-caller
   ```

2. **Ensure that all the dependencies are installed**.

3. **Configure the system prompt**: The default system prompt simulates a sales conversation and is included in the code, but you can customize it by modifying the `system_prompt` argument when initializing the `AIColdCaller` class.

4. **Run the AI Cold Caller**:

   ```bash
   python ai_cold_caller.py
   ```

5. **Logging**: The system logs all interactions, analytics, and audio recordings in the `call_logs` directory by default. You can modify the log directory path by passing the `log_dir` parameter during initialization.

## Workflow

1. **Initialization**: The system initializes all necessary models and logging functionality.
2. **Audio Recording**: The system records the audio from the microphone and saves it as a `.wav` file.
3. **Speech-to-Text**: The Whisper model transcribes the recorded audio into text.
4. **Response Generation**: The language model generates a response based on the conversation history and the customer's sentiment.
5. **Text-to-Speech**: The system converts the generated response into speech and plays it back to the customer.
6. **Analytics**: Throughout the conversation, the system tracks analytics like call duration, agent talk time, sentiment, and response types (positive/negative/neutral).
7. **Logging**: The conversation and analytics are saved in the `call_logs` directory, including transcripts and a summary of the call analytics.

## Analytics Data

The following metrics are tracked:

- **Call Duration**: Total duration of the call.
- **Customer Talk Time**: Time the customer spent speaking.
- **Agent Talk Time**: Time the agent spent speaking.
- **Turn Count**: Number of interactions (turns) during the conversation.
- **Positive Responses**: Number of positive responses from the customer.
- **Negative Responses**: Number of negative responses from the customer.
- **Sentiment Scores**: Sentiment scores for each interaction (0 for negative, 1 for positive).
- **Average Sentiment**: The average sentiment score for the entire call.

## Example of System Output

```text
AI Cold Caller started. Press Ctrl+C to exit.
Listening...
Audio recorded and saved to: call_logs/recordings/recording_20231231_123456.wav
Customer: "Hello, I wanted to ask about your software"
Agent: "Hi! This is [Your Name] from [Company]. Have I caught you at a good time?"
...
```

## File Structure

```
.
├── ai_cold_caller.py        # Main script to run the AI Cold Caller
└── call_logs/               # Directory where logs, recordings, and transcripts are stored
    ├── recordings/          # Directory for saved audio recordings
    ├── transcripts/         # Directory for saved conversation transcripts
    ├── analytics/           # Directory for saved analytics (JSON files)
    └── cold_caller.log      # Log file for all system messages and errors
```



