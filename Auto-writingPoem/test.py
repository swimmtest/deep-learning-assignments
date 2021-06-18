import sys
import torch
from opts import Config
from dataset import get_data
from model import PoetryModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()

def generate_poetry_by_start(model, start_words, word2idx, idx2word, prefix_words=None):

    results = list(start_words)
    start_words_length = len(start_words)
    # set first word is <START>
    input = torch.Tensor([word2idx["<START>"]]).view(1, 1).long()
    hidden = None

    # prefix_words guide poetry firstly
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2idx[word]]).view(1, 1)
    
    for index in range(cfg.generate_max_length_limit):
        gen_word = ""
        output, hidden = model(input, hidden)
        if index < start_words_length:
            word = results[index]
            input = input.data.new([word2idx[word]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            gen_word = idx2word[top_index]
            results.append(gen_word)
            input = input.data.new([top_index]).view(1, 1)
        
        if gen_word == "<EOP>":
            del results[-1]
            break
    
    return results

def generate_acrostic(model, start_words, word2idx, idx2word, prefix_words):
    results = []
    start_words_length = len(start_words)
    input = (torch.Tensor([word2idx["<START>"]]).view(1,1).long())

    hidden = None
    index = 0
    pre_word = "<START>"

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2idx[word]])).view(1,1)

    for i in range(cfg.generate_max_length_limit):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        word = idx2word[top_index]

        if pre_word in {u"。", u"！", "<START>"}:
            if index == start_words_length: break
            else:
                word = start_words[index]
                index += 1
                input = (input.data.new([word2idx[word]])).view(1, 1)
        else:
            input = (input.data.new([word2idx[word]])).view(1, 1)
        
        results.append(word)
        pre_word = word

    return results    

def generate(**kwargs):
    for para, value in kwargs.items(): setattr(cfg, para, value)
    data, word2idx, idx2word = get_data(cfg)
    model = PoetryModel(len(word2idx), 128, 256)
    map_location = lambda s, l: s
    model.load_state_dict(torch.load(cfg.pre_train_model_path, map_location=map_location))

    # python2 and python3 str compatibility
    if sys.version_info.major == 3:
        if cfg.start_words.isprintable():
            start_words = cfg.start_words
            prefix_words = cfg.prefix_words if cfg.start_words else None
        else:
            start_words = cfg.start_words.encode("ascii", "surrogateescape").decode("utf-8")
            prefix_words = cfg.prefix_words.encode("ascii", "surrogateescape").decode("utf-8") if cfg.prefix_words else None
    else:
        start_words = cfg.start_words.decode("utf-8")
        prefix_words = cfg.prefix_words.decode("utf-8") if cfg.prefix_words else None

    start_words = start_words.replace(",", u"，").replace(".", u"。").replace("?", u"？")
    gen_poetry = generate_acrostic if cfg.acrostic else generate_poetry_by_start

    result = gen_poetry(model, start_words, word2idx, idx2word, prefix_words)
    print("".join(result))


if __name__ == '__main__':
    import fire
    fire.Fire()