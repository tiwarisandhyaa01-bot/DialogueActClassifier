import pandas as pd
import random
import os

# Dialogue Act Categories
dialogue_acts = [
    "Greeting",
    "Question", 
    "Answer",
    "Complaint",
    "Request",
    "Acknowledgment",
    "Closing"
]

# Realistic customer support templates for each dialogue act
templates = {
    "Greeting": [
        "Hello! How can I help you today?",
        "Good morning! Welcome to our support center.",
        "Hi there! Thanks for reaching out to us.",
        "Greetings! I'm here to assist you.",
        "Hello, welcome! What can I do for you today?",
        "Hi! Thank you for contacting our support team.",
        "Good afternoon! How may I assist you?",
        "Hey! I'm happy to help you with your issue.",
        "Hello! Thanks for getting in touch with us.",
        "Hi, I'm your support agent. How can I help?"
    ],
    
    "Question": [
        "What is your order number?",
        "Can you provide your account email address?",
        "When did you first notice this issue?",
        "Have you tried restarting the application?",
        "Which payment method did you use?",
        "What error message are you seeing?",
        "Can you describe the problem in more detail?",
        "Is this issue happening on mobile or desktop?",
        "Did you receive a confirmation email?",
        "What browser are you currently using?",
        "Could you tell me when you placed the order?",
        "Have you updated to the latest version?",
        "What happens when you click the submit button?",
        "Can you share a screenshot of the error?",
        "Which product are you having trouble with?"
    ],
    
    "Answer": [
        "Your order was shipped on December 20th.",
        "The refund will be processed within 5-7 business days.",
        "Your account has been successfully verified.",
        "The issue was caused by a temporary server outage.",
        "You can track your package using tracking number XYZ123.",
        "Our premium plan costs $29.99 per month.",
        "The password reset link has been sent to your email.",
        "Your subscription will renew on January 15th.",
        "We currently support payments via credit card and PayPal.",
        "The software is compatible with Windows 10 and above.",
        "Your order total is $156.50 including taxes.",
        "The delivery estimate is 3-5 business days.",
        "You have 2 remaining credits in your account.",
        "The feature you requested is available in our Pro plan.",
        "Your support ticket has been assigned ID #45621."
    ],
    
    "Complaint": [
        "I've been waiting for my order for 3 weeks now!",
        "This is the third time I'm contacting support about this issue.",
        "The product I received is completely different from what I ordered.",
        "Your website keeps crashing whenever I try to make a payment.",
        "I was charged twice for the same transaction!",
        "The customer service has been extremely unhelpful so far.",
        "I still haven't received my refund after 2 months.",
        "The quality of the product is very poor and not as described.",
        "I'm very disappointed with the service I've received.",
        "This app is full of bugs and constantly freezes.",
        "I've been on hold for over an hour trying to reach someone.",
        "The delivery was late and the package was damaged.",
        "I never authorized this subscription renewal!",
        "Your system deleted all my saved data without warning.",
        "I'm extremely frustrated with how long this is taking."
    ],
    
    "Request": [
        "I would like to cancel my subscription please.",
        "Can you please expedite the shipping on my order?",
        "I need to update my billing address.",
        "Could you please resend the confirmation email?",
        "I want to upgrade to the premium plan.",
        "Please refund my payment as soon as possible.",
        "I'd like to speak with a supervisor about this matter.",
        "Can you please remove my account from your database?",
        "I need help resetting my password.",
        "Could you please check the status of my refund?",
        "I would like to return this product for a full refund.",
        "Please add an extra item to my existing order.",
        "Can you extend my trial period by one week?",
        "I need assistance setting up my new account.",
        "Could you please send me an invoice for my purchase?"
    ],
    
    "Acknowledgment": [
        "Thank you for the information.",
        "Okay, I understand now.",
        "Got it, thanks for clarifying.",
        "That makes sense, thank you.",
        "Alright, I'll try that solution.",
        "Perfect, I appreciate your help.",
        "Understood, I'll wait for the update.",
        "Thanks, that's exactly what I needed to know.",
        "Okay, I'll check my email for the confirmation.",
        "Great, thank you for looking into this.",
        "I see, thank you for explaining that.",
        "Alright, I'll follow those steps.",
        "Thanks for the quick response!",
        "Okay, I'll keep that in mind.",
        "Noted, thank you for your assistance."
    ],
    
    "Closing": [
        "Thank you for your help! Have a great day.",
        "Thanks, that solved my issue. Goodbye!",
        "I appreciate your assistance. Take care!",
        "Thank you, you've been very helpful!",
        "Great, thanks! That's all I needed.",
        "Perfect, have a wonderful day!",
        "Thanks for resolving this so quickly!",
        "I appreciate your time. Goodbye!",
        "Thank you, I'm all set now!",
        "Thanks again for your help!",
        "That's everything I needed, thank you!",
        "Great service, thank you!",
        "Thanks, I really appreciate it!",
        "Perfect, have a nice day!",
        "Thank you so much for your help!"
    ]
}

def generate_dataset(num_samples=600):
    """
    Generates a balanced synthetic dataset for dialogue act classification
    """
    data = []
    samples_per_class = num_samples // len(dialogue_acts)
    
    for act in dialogue_acts:
        act_templates = templates[act]
        
        # Generate samples for this dialogue act
        for _ in range(samples_per_class):
            # Pick a random template
            text = random.choice(act_templates)
            
            # Add slight variations (optional)
            if random.random() > 0.7:  # 30% chance of variation
                variations = {
                    "!": ["!", ".", ""],
                    "?": ["?", ""],
                    "please": ["please", "kindly", ""],
                    "Hello": ["Hello", "Hi", "Hey"],
                    "Thank you": ["Thank you", "Thanks", "Thank you so much"]
                }
                
                for original, replacements in variations.items():
                    if original in text:
                        text = text.replace(original, random.choice(replacements))
            
            data.append({
                "text": text.strip(),
                "dialogue_act": act
            })
    
    # Shuffle the dataset
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def main():
    """
    Main function to generate and save the dataset
    """
    print("=" * 60)
    print("DIALOGUE ACT DATASET GENERATOR")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    print("\n[1/3] Generating synthetic customer support dialogues...")
    df = generate_dataset(num_samples=600)
    
    # Display statistics
    print(f"\n[2/3] Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Dialogue acts: {df['dialogue_act'].nunique()}")
    print(f"\n   Class Distribution:")
    print(df['dialogue_act'].value_counts().to_string())
    
    # Save to CSV
    output_path = "data/customer_support_dialogues.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[3/3] Dataset saved to: {output_path}")
    
    # Display sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 60)
    print(df.head().to_string(index=False))
    
    print("\n✅ Dataset generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()