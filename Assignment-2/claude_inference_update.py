# ============================================================================
# FIXED INFERENCE FUNCTIONS - Multi-Word Support
# Handles space-separated words correctly
# ============================================================================

import torch
import re

# ============================================================================
# HELPER FUNCTION: Process Multi-Word Input
# ============================================================================

def process_multi_word_input(text):
    """
    Split input text into words while preserving punctuation and spaces
    Returns list of words and their positions
    """
    # Split by whitespace but keep track of original spacing
    words = text.strip().split()
    return words

def join_transliterated_words(transliterated_words):
    """Join transliterated words with spaces"""
    return ' '.join(transliterated_words)

# ============================================================================
# LSTM INFERENCE FUNCTIONS (FIXED)
# ============================================================================

def greedy_decode_lstm_single_word(model, src, src_vocab, tgt_vocab, device, max_len=50):
    """Greedy decoding for LSTM - SINGLE WORD ONLY"""
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_vocab.char2idx['<SOS>']] + 
                                 src_vocab.encode(src) + 
                                 [src_vocab.char2idx['<EOS>']]).unsqueeze(0).to(device)
        
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        outputs = [tgt_vocab.char2idx['<SOS>']]
        
        for _ in range(max_len):
            x = torch.tensor([outputs[-1]]).unsqueeze(0).to(device)
            output, hidden, cell, _ = model.decoder(x, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            outputs.append(pred_token)
            if pred_token == tgt_vocab.char2idx['<EOS>']:
                break
        
        return tgt_vocab.decode(outputs)

def greedy_decode_lstm(model, text, src_vocab, tgt_vocab, device, max_len=50):
    """
    Greedy decoding for LSTM - MULTI-WORD SUPPORT
    Handles space-separated words correctly
    """
    # Handle empty input
    if not text or not text.strip():
        return ""
    
    # Split into words
    words = process_multi_word_input(text)
    
    # Transliterate each word separately
    transliterated_words = []
    for word in words:
        if word.strip():  # Skip empty strings
            result = greedy_decode_lstm_single_word(model, word.strip(), src_vocab, tgt_vocab, device, max_len)
            transliterated_words.append(result)
    
    # Join with spaces
    return join_transliterated_words(transliterated_words)

def beam_search_lstm_single_word(model, src, src_vocab, tgt_vocab, device, beam_width=5, max_len=50):
    """Beam search decoding for LSTM - SINGLE WORD ONLY"""
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_vocab.char2idx['<SOS>']] + 
                                 src_vocab.encode(src) + 
                                 [src_vocab.char2idx['<EOS>']]).unsqueeze(0).to(device)
        
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Initialize beam
        sequences = [([tgt_vocab.char2idx['<SOS>']], 0.0, hidden, cell)]
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score, h, c in sequences:
                if seq[-1] == tgt_vocab.char2idx['<EOS>']:
                    all_candidates.append((seq, score, h, c))
                    continue
                
                x = torch.tensor([seq[-1]]).unsqueeze(0).to(device)
                output, new_h, new_c, _ = model.decoder(x, h, c, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1)
                
                top_k = torch.topk(log_probs, beam_width)
                
                for prob, idx in zip(top_k.values[0], top_k.indices[0]):
                    candidate_seq = seq + [idx.item()]
                    candidate_score = score + prob.item()
                    all_candidates.append((candidate_seq, candidate_score, new_h, new_c))
            
            # Select top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
            
            # Check if all sequences ended
            if all(seq[-1] == tgt_vocab.char2idx['<EOS>'] for seq, _, _, _ in sequences):
                break
        
        best_sequence = sequences[0][0]
        return tgt_vocab.decode(best_sequence)

def beam_search_lstm(model, text, src_vocab, tgt_vocab, device, beam_width=5, max_len=50):
    """
    Beam search decoding for LSTM - MULTI-WORD SUPPORT
    Handles space-separated words correctly
    """
    # Handle empty input
    if not text or not text.strip():
        return ""
    
    # Split into words
    words = process_multi_word_input(text)
    
    # Transliterate each word separately
    transliterated_words = []
    for word in words:
        if word.strip():  # Skip empty strings
            result = beam_search_lstm_single_word(model, word.strip(), src_vocab, tgt_vocab, device, beam_width, max_len)
            transliterated_words.append(result)
    
    # Join with spaces
    return join_transliterated_words(transliterated_words)

# ============================================================================
# TRANSFORMER INFERENCE FUNCTIONS (FIXED)
# ============================================================================

def greedy_decode_transformer_single_word(model, src, src_vocab, tgt_vocab, device, max_len=50):
    """Greedy decoding for Transformer - SINGLE WORD ONLY"""
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_vocab.char2idx['<SOS>']] + 
                                 src_vocab.encode(src) + 
                                 [src_vocab.char2idx['<EOS>']]).unsqueeze(0).to(device)
        
        outputs = [tgt_vocab.char2idx['<SOS>']]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor(outputs).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor)
            pred_token = output[0, -1].argmax().item()
            outputs.append(pred_token)
            if pred_token == tgt_vocab.char2idx['<EOS>']:
                break
        
        return tgt_vocab.decode(outputs)

def greedy_decode_transformer(model, text, src_vocab, tgt_vocab, device, max_len=50):
    """
    Greedy decoding for Transformer - MULTI-WORD SUPPORT
    Handles space-separated words correctly
    """
    # Handle empty input
    if not text or not text.strip():
        return ""
    
    # Split into words
    words = process_multi_word_input(text)
    
    # Transliterate each word separately
    transliterated_words = []
    for word in words:
        if word.strip():  # Skip empty strings
            result = greedy_decode_transformer_single_word(model, word.strip(), src_vocab, tgt_vocab, device, max_len)
            transliterated_words.append(result)
    
    # Join with spaces
    return join_transliterated_words(transliterated_words)

def beam_search_transformer_single_word(model, src, src_vocab, tgt_vocab, device, beam_width=5, max_len=50):
    """Beam search decoding for Transformer - SINGLE WORD ONLY"""
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([src_vocab.char2idx['<SOS>']] + 
                                 src_vocab.encode(src) + 
                                 [src_vocab.char2idx['<EOS>']]).unsqueeze(0).to(device)
        
        # Initialize beam
        sequences = [([tgt_vocab.char2idx['<SOS>']], 0.0)]
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score in sequences:
                if seq[-1] == tgt_vocab.char2idx['<EOS>']:
                    all_candidates.append((seq, score))
                    continue
                
                tgt_tensor = torch.tensor(seq).unsqueeze(0).to(device)
                output = model(src_tensor, tgt_tensor)
                log_probs = torch.log_softmax(output[0, -1], dim=0)
                
                top_k = torch.topk(log_probs, beam_width)
                
                for prob, idx in zip(top_k.values, top_k.indices):
                    candidate_seq = seq + [idx.item()]
                    candidate_score = score + prob.item()
                    all_candidates.append((candidate_seq, candidate_score))
            
            # Select top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
            
            # Check if all sequences ended
            if all(seq[-1] == tgt_vocab.char2idx['<EOS>'] for seq, _ in sequences):
                break
        
        best_sequence = sequences[0][0]
        return tgt_vocab.decode(best_sequence)

def beam_search_transformer(model, text, src_vocab, tgt_vocab, device, beam_width=5, max_len=50):
    """
    Beam search decoding for Transformer - MULTI-WORD SUPPORT
    Handles space-separated words correctly
    """
    # Handle empty input
    if not text or not text.strip():
        return ""
    
    # Split into words
    words = process_multi_word_input(text)
    
    # Transliterate each word separately
    transliterated_words = []
    for word in words:
        if word.strip():  # Skip empty strings
            result = beam_search_transformer_single_word(model, word.strip(), src_vocab, tgt_vocab, device, beam_width, max_len)
            transliterated_words.append(result)
    
    # Join with spaces
    return join_transliterated_words(transliterated_words)

# ============================================================================
# UPDATED EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, model_type, decode_type, test_data, src_vocab, tgt_vocab, device, num_samples=1000):
    """
    Evaluate model on test data - FIXED VERSION
    Handles multi-word inputs correctly
    """
    from tqdm import tqdm
    import random
    
    print(f"\n🔍 Evaluating {model_type} with {decode_type} decoding...")
    
    # Sample test data
    test_sample = random.sample(test_data, min(num_samples, len(test_data)))
    
    predictions = []
    references = []
    
    for item in tqdm(test_sample, desc=f"Evaluating"):
        src = item['source']
        ref = item['target']
        
        if model_type == "LSTM":
            if decode_type == "Greedy":
                pred = greedy_decode_lstm(model, src, src_vocab, tgt_vocab, device)
            else:  # Beam Search
                pred = beam_search_lstm(model, src, src_vocab, tgt_vocab, device)
        else:  # Transformer
            if decode_type == "Greedy":
                pred = greedy_decode_transformer(model, src, src_vocab, tgt_vocab, device)
            else:  # Beam Search
                pred = beam_search_transformer(model, src, src_vocab, tgt_vocab, device)
        
        predictions.append(pred)
        references.append(ref)
    
    # Calculate metrics
    def word_accuracy(predictions, references):
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    def char_f1_score(prediction, reference):
        pred_chars = list(prediction)
        ref_chars = list(reference)
        
        if len(pred_chars) == 0 and len(ref_chars) == 0:
            return 1.0
        if len(pred_chars) == 0 or len(ref_chars) == 0:
            return 0.0
        
        common = 0
        for i, (p, r) in enumerate(zip(pred_chars, ref_chars)):
            if p == r:
                common += 1
        
        precision = common / len(pred_chars) if len(pred_chars) > 0 else 0
        recall = common / len(ref_chars) if len(ref_chars) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def average_char_f1(predictions, references):
        scores = [char_f1_score(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0
    
    word_acc = word_accuracy(predictions, references)
    char_f1 = average_char_f1(predictions, references)
    
    return word_acc, char_f1, predictions, references

# ============================================================================
# UPDATED GUI FUNCTIONS
# ============================================================================

def transliterate(text, model_choice, decode_choice):
    """
    Transliterate text using selected model and decoding strategy
    FIXED VERSION - Handles multi-word inputs
    """
    if not text.strip():
        return "Please enter some text!"
    
    text = text.strip().lower()
    
    try:
        if model_choice == "LSTM":
            if decode_choice == "Greedy":
                result = greedy_decode_lstm(lstm_model, text, src_vocab, tgt_vocab, device)
            else:  # Beam Search
                result = beam_search_lstm(lstm_model, text, src_vocab, tgt_vocab, device, beam_width=5)
        else:  # Transformer
            if decode_choice == "Greedy":
                result = greedy_decode_transformer(transformer_model, text, src_vocab, tgt_vocab, device)
            else:  # Beam Search
                result = beam_search_transformer(transformer_model, text, src_vocab, tgt_vocab, device, beam_width=5)
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def compare_all_models(text):
    """
    Compare all model and decoding combinations
    FIXED VERSION - Handles multi-word inputs
    """
    if not text.strip():
        return "Please enter some text!", "", "", ""
    
    text = text.strip().lower()
    
    try:
        lstm_greedy = greedy_decode_lstm(lstm_model, text, src_vocab, tgt_vocab, device)
        lstm_beam = beam_search_lstm(lstm_model, text, src_vocab, tgt_vocab, device, beam_width=5)
        trans_greedy = greedy_decode_transformer(transformer_model, text, src_vocab, tgt_vocab, device)
        trans_beam = beam_search_transformer(transformer_model, text, src_vocab, tgt_vocab, device, beam_width=5)
        
        return lstm_greedy, lstm_beam, trans_greedy, trans_beam
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg

# ============================================================================
# TESTING THE FIXED INFERENCE
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING FIXED INFERENCE WITH MULTI-WORD SUPPORT")
    print("="*60)
    
    # Test examples with multiple words
    multi_word_examples = [
        "namaste dosto",
        "mujhe hindi sikhna hai",
        "main ghar ja raha hoon",
        "aap kaise hain",
        "subh prabhat aur shubh ratri",
        "bharat ek mahan desh hai"
    ]
    
    print("\n🔹 Testing LSTM with Greedy Decoding:")
    for example in multi_word_examples:
        result = greedy_decode_lstm(lstm_model, example, src_vocab, tgt_vocab, device)
        print(f"  Input:  {example}")
        print(f"  Output: {result}")
        print()
    
    print("\n🔹 Testing LSTM with Beam Search:")
    for example in multi_word_examples:
        result = beam_search_lstm(lstm_model, example, src_vocab, tgt_vocab, device)
        print(f"  Input:  {example}")
        print(f"  Output: {result}")
        print()
    
    print("\n🔹 Testing Transformer with Greedy Decoding:")
    for example in multi_word_examples:
        result = greedy_decode_transformer(transformer_model, example, src_vocab, tgt_vocab, device)
        print(f"  Input:  {example}")
        print(f"  Output: {result}")
        print()
    
    print("\n🔹 Testing Transformer with Beam Search:")
    for example in multi_word_examples:
        result = beam_search_transformer(transformer_model, example, src_vocab, tgt_vocab, device)
        print(f"  Input:  {example}")
        print(f"  Output: {result}")
        print()
    
    print("✅ All tests completed!")
    print("="*60)

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
USAGE:

1. Replace your old inference functions with these fixed versions

2. For single model inference:
   result = greedy_decode_lstm(model, "mujhe hindi sikhna hai", src_vocab, tgt_vocab, device)
   # Output: "मुझे हिंदी सीखना है" (with spaces preserved)

3. For beam search:
   result = beam_search_transformer(model, "namaste dosto", src_vocab, tgt_vocab, device, beam_width=5)
   # Output: "नमस्ते दोस्तो" (with spaces preserved)

4. The GUI functions are also updated to handle multi-word inputs correctly

5. Evaluation function now works correctly with multi-word test samples

KEY CHANGES:
- Each word in the input is transliterated separately
- Results are joined with spaces
- Empty strings are handled gracefully
- Works seamlessly with existing model architecture
"""