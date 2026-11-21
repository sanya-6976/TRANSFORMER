import torch 

def predict_next_token(logits):

    # Select logits from the final sequence position
    next_token_logits = logits[:,-1,:]

    # Select the vocabulary token with the highest score
    next_token = torch.argmax(
         next_token_logits , dim= -1
    )

    return next_token

def generate_token(model , token_ids , max_new_tokens):

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions for the current sequence
            logits = model(token_ids)
            # Predict the next token
            next_token = predict_next_token(logits)

           # Append predicted token to the sequence
            next_token = next_token.unsqueeze(1)

            token_ids = torch.cat(
                (token_ids, next_token),
                dim=1
            )

    return token_ids 


