import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

# Define a simple Named Entity Recognition (NER) model for entity extraction
class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

# Define a simple relation extraction model
class RelationExtractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations):
        super(RelationExtractionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_relations)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# Define a typing model for entity classification
class TypingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_types):
        super(TypingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_types)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# Utility function to construct a knowledge graph
def construct_knowledge_graph(entities: List[str], relations: List[Tuple[str, str, str]]) -> Dict:
    graph = defaultdict(list)
    for subj, rel, obj in relations:
        graph[subj].append((rel, obj))
    return graph

# Dummy data for testing
def generate_dummy_data():
    vocab_size = 100
    seq_len = 10
    batch_size = 2
    num_classes = 5
    num_relations = 3
    num_types = 4

    # Random input data
    input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    entity_labels = torch.randint(0, num_classes, (batch_size, seq_len))
    relation_data = torch.rand(batch_size, seq_len, seq_len)
    relation_labels = torch.randint(0, num_relations, (batch_size, seq_len, seq_len))
    entity_embeddings = torch.rand(batch_size, seq_len, 50)
    type_labels = torch.randint(0, num_types, (batch_size, seq_len))

    return input_data, entity_labels, relation_data, relation_labels, entity_embeddings, type_labels

if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 100
    embedding_dim = 50
    hidden_dim = 64
    num_classes = 5
    num_relations = 3
    num_types = 4
    learning_rate = 0.001
    epochs = 5

    # Generate dummy data
    input_data, entity_labels, relation_data, relation_labels, entity_embeddings, type_labels = generate_dummy_data()

    # Initialize models
    ner_model = NERModel(vocab_size, embedding_dim, hidden_dim, num_classes)
    relation_model = RelationExtractionModel(50, hidden_dim, num_relations)
    typing_model = TypingModel(50, hidden_dim, num_types)

    # Define loss functions and optimizers
    criterion_ner = nn.CrossEntropyLoss()
    criterion_relation = nn.CrossEntropyLoss()
    criterion_typing = nn.CrossEntropyLoss()

    optimizer_ner = optim.Adam(ner_model.parameters(), lr=learning_rate)
    optimizer_relation = optim.Adam(relation_model.parameters(), lr=learning_rate)
    optimizer_typing = optim.Adam(typing_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Train NER model
        ner_model.train()
        optimizer_ner.zero_grad()
        ner_logits = ner_model(input_data)
        ner_loss = criterion_ner(ner_logits.view(-1, num_classes), entity_labels.view(-1))
        ner_loss.backward()
        optimizer_ner.step()

        # Train relation extraction model
        relation_model.train()
        optimizer_relation.zero_grad()
        relation_logits = relation_model(relation_data.view(-1, 50))
        relation_loss = criterion_relation(relation_logits, relation_labels.view(-1))
        relation_loss.backward()
        optimizer_relation.step()

        # Train typing model
        typing_model.train()
        optimizer_typing.zero_grad()
        typing_logits = typing_model(entity_embeddings.view(-1, 50))
        typing_loss = criterion_typing(typing_logits, type_labels.view(-1))
        typing_loss.backward()
        optimizer_typing.step()

        print(f"Epoch {epoch + 1}/{epochs}, NER Loss: {ner_loss.item()}, Relation Loss: {relation_loss.item()}, Typing Loss: {typing_loss.item()}")

    # Dummy inference
    ner_model.eval()
    relation_model.eval()
    typing_model.eval()

    with torch.no_grad():
        ner_predictions = torch.argmax(ner_model(input_data), dim=-1)
        relation_predictions = torch.argmax(relation_model(relation_data.view(-1, 50)), dim=-1).view(relation_data.shape[:-1])
        typing_predictions = torch.argmax(typing_model(entity_embeddings.view(-1, 50)), dim=-1).view(entity_embeddings.shape[:-1])

    # Construct a dummy knowledge graph
    entities = [f"Entity_{i}" for i in range(5)]
    relations = [("Entity_0", "related_to", "Entity_1"), ("Entity_2", "causes", "Entity_3")]
    knowledge_graph = construct_knowledge_graph(entities, relations)

    print("Knowledge Graph:", knowledge_graph)