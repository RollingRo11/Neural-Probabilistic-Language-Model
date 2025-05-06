
from model import BigramModel

def main():
    model = BigramModel('names.txt')
    
    print(f"dataset contains {len(model.words)} words")
    print(f"Shortest word length: {min(len(w) for w in model.words)}")
    print(f"Longest word length: {max(len(w) for w in model.words)}")
    
    print("\ntraining...")
    model.train(num_iterations=350)
    
    print("\nmaking words:")
    names = model.generate_names(num_names=10)
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
   
if __name__ == "__main__":
    main()
