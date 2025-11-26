"""
Sample Data Generator for NLP Text Classification Dashboard
Generates realistic test data for the Streamlit app
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# Sample transcripts by category
SAMPLE_TRANSCRIPTS = {
    "login issue": [
        "Consumer: I can't log into my account, it says wrong password | Agent: Let me help you reset it | 2024-01-15",
        "Consumer: My account is locked after too many attempts | Agent: I'll unlock it for you",
        "Consumer: The verification code is not working | Agent: Let me send you a new code",
        "Consumer: I forgot my password and can't reset it | Agent: I can help with that",
        "Consumer: Two-factor authentication is not working | Agent: Let's troubleshoot that",
    ],
    "playback issue": [
        "Consumer: Songs keep stopping in the middle | Agent: That's frustrating, let me check | 2024-01-16",
        "Consumer: Music won't play at all, just buffers | Agent: Let's check your connection",
        "Consumer: Audio quality is really poor | Agent: I'll look into that",
        "Consumer: The app keeps pausing randomly | Agent: That shouldn't happen",
        "Consumer: No sound coming from the speakers | Agent: Let's diagnose this",
    ],
    "subscription issue": [
        "Consumer: I was charged twice for premium | Agent: I'll process a refund right away | 2024-01-17",
        "Consumer: Want to cancel my subscription | Agent: I can help you with that",
        "Consumer: How do I upgrade to family plan? | Agent: Let me explain the options",
        "Consumer: My payment failed but it worked before | Agent: Let's update your payment method",
        "Consumer: Need a refund for an accidental charge | Agent: I'll take care of that",
    ],
    "device issue": [
        "Consumer: Bluetooth connection keeps dropping | Agent: Let's try reconnecting | 2024-01-18",
        "Consumer: Can't connect to my CarPlay | Agent: I'll help you set that up",
        "Consumer: Alexa integration stopped working | Agent: Let's relink your account",
        "Consumer: The app crashes on my Android phone | Agent: What Android version do you have?",
        "Consumer: Chromecast not showing up | Agent: Let's troubleshoot the connection",
    ],
    "network failure": [
        "Consumer: Keep getting connection error | Agent: Let's check the server status | 2024-01-19",
        "Consumer: Says I'm offline but I have internet | Agent: That's odd, let me investigate",
        "Consumer: Spotify down? Can't connect | Agent: Let me verify the server status",
        "Consumer: Timeout error every time I try | Agent: This might be a network issue",
    ],
    "app crash": [
        "Consumer: App crashes when I open playlists | Agent: What device are you using? | 2024-01-20",
        "Consumer: The app freezes and I have to restart | Agent: That shouldn't happen",
        "Consumer: Getting error message repeatedly | Agent: What's the error code?",
        "Consumer: App is so slow it's unusable | Agent: Let's clear the cache",
    ],
    "general feedback": [
        "Consumer: Love the new interface! | Agent: Great to hear! | 2024-01-21",
        "Consumer: Can you add a sleep timer feature? | Agent: I'll pass that to the team",
        "Consumer: The app is amazing, just wanted to say thanks | Agent: Thank you for the feedback!",
        "Consumer: Suggestion - dark mode for desktop | Agent: That's a popular request",
    ],
}

def generate_sample_data(num_rows: int = 100, output_format: str = "csv") -> str:
    """
    Generate sample transcript data for testing.
    
    Args:
        num_rows: Number of sample rows to generate
        output_format: Output format ('csv' or 'xlsx')
    
    Returns:
        Filename of generated file
    """
    data = []
    categories = list(SAMPLE_TRANSCRIPTS.keys())
    
    for i in range(num_rows):
        # Random category
        category = random.choice(categories)
        
        # Random transcript from that category
        transcript = random.choice(SAMPLE_TRANSCRIPTS[category])
        
        # Generate conversation ID
        conv_id = f"CONV_{str(i+1).zfill(5)}"
        
        data.append({
            "Conversation Id": conv_id,
            "transcripts": transcript
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format.lower() == "csv":
        filename = f"sample_data_{num_rows}rows_{timestamp}.csv"
        df.to_csv(filename, index=False)
    else:
        filename = f"sample_data_{num_rows}rows_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
    
    return filename


def main():
    """Main function to generate sample data"""
    print("🎲 Sample Data Generator")
    print("=" * 50)
    print()
    
    # Get user input
    try:
        num_rows = input("How many rows to generate? (default: 100): ").strip()
        num_rows = int(num_rows) if num_rows else 100
        
        if num_rows < 1 or num_rows > 10000:
            print("❌ Please enter a number between 1 and 10,000")
            return
        
        format_choice = input("Output format (csv/xlsx)? (default: csv): ").strip().lower()
        format_choice = format_choice if format_choice in ["csv", "xlsx"] else "csv"
        
        print()
        print(f"📝 Generating {num_rows} sample rows...")
        
        filename = generate_sample_data(num_rows, format_choice)
        
        print(f"✅ Sample data generated: {filename}")
        print()
        print("📊 Category distribution:")
        df = pd.read_csv(filename) if format_choice == "csv" else pd.read_excel(filename)
        
        # Show distribution (approximate)
        categories = list(SAMPLE_TRANSCRIPTS.keys())
        print(f"   - Random mix of {len(categories)} categories")
        print(f"   - Approximately {num_rows // len(categories)} rows per category")
        print()
        print("🚀 You can now upload this file to the Streamlit app!")
        
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
