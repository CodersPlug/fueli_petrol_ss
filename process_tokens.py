import pandas as pd
import tiktoken
import json
from datetime import datetime
from typing import List, Dict, Any

def load_csv(file_path: str) -> pd.DataFrame:
    """Load and preprocess the CSV file."""
    # Read CSV with proper decimal and thousands separators
    df = pd.read_csv(file_path, 
                    thousands='.', 
                    decimal=',',
                    parse_dates=False)  # Don't parse dates automatically
    
    # Convert date string to datetime manually
    df['Fecha y Hora'] = pd.to_datetime(df['Fecha y Hora'], 
                                      format='%d/%m/%Y %H:%M:%S',
                                      errors='coerce')
    return df

def format_row(row: pd.Series) -> str:
    """Format a row into a meaningful text representation."""
    try:
        return (
            f"En {row['Fecha y Hora'].strftime('%d/%m/%Y %H:%M:%S')}, "
            f"en el pico {row['Pico']}, "
            f"se despachó {row['Producto']}, "
            f"por un importe de ${row['Importe']:.2f}, "
            f"con un volumen de {row['Volumen']:.2f} litros "
            f"a un precio por unidad de ${row['PPU']:.2f}"
        )
    except Exception as e:
        print(f"Error formatting row: {e}")
        print(f"Row data: {row.to_dict()}")
        return None

def tokenize_text(text: str, tokenizer: Any) -> List[int]:
    """Tokenize text using the specified tokenizer."""
    if text is None:
        return []
    return tokenizer.encode(text)

def main():
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    print("Loading CSV file...")
    df = load_csv("Petrol - Despachos.csv")
    
    print("Processing and tokenizing data...")
    tokenized_data = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}...")
            
        try:
            # Format the row into text
            text = format_row(row)
            if text is None:
                continue
                
            # Tokenize the text
            tokens = tokenize_text(text, tokenizer)
            
            # Store both the original text and its tokens
            tokenized_data.append({
                "text": text,
                "tokens": tokens,
                "n_tokens": len(tokens)
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    
    print("Saving tokenized data...")
    with open("tokenized_data.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
    
    print("\nDone! Summary:")
    print(f"Total records processed: {len(tokenized_data)}")
    total_tokens = sum(item["n_tokens"] for item in tokenized_data)
    print(f"Total tokens: {total_tokens}")
    if tokenized_data:
        print(f"Average tokens per record: {total_tokens / len(tokenized_data):.2f}")
    print("\nSample tokenized text:")
    if tokenized_data:
        print(tokenized_data[0]["text"])
        print(f"Number of tokens: {tokenized_data[0]['n_tokens']}")

if __name__ == "__main__":
    main() 