class Config(object):
    # The path of poetry tang/song dataset, default tang
    dataset_path = "data/song/"	
    # Binary files after pre-processing can be used directly for model train
    pickle_file_path = "data/train_song_pickle.npz"	
    # Author limit, if not None, will only learn the author's verse
    author_limit = None	    
    # length limit, if it is not None, only the verses of the specified length will be learned.
    length_limit = None	    
    # class limit, value choose[poet.tang, poet.song]
    class_limit = "poet.song"	
    # The model learning rate
    learning_rate = 1e-3	
    weight_decay = 1e-4
    # The model train epoch
    epoch = 20	  
    # model train batch size          
    batch_size = 128	    
	# The part after the sentence that exceeds this length is discarded,
	# and the sentence smaller than this length is padding at the specified position.
    max_length = 200
    plot_every = 20
    # if or not use visodm
    use_env = False	 
    # visdom env       
    env = 'poetry'
    # generate poetry max length	        
    generate_max_length_limit = 200	
    debug_path = "p_debug"
    # The path of pre-train model
    pre_train_model_path = None # "checkpoints/song_json/poet.song_20.pth"
    # Control poetry        	
    prefix_words = "閑云潭影日悠悠。"	
    # poetry start
    start_words = "湖光秋月兩相和"	
    # Is it a Tibetan poem?
    acrostic = False	    
    model_prefix = "checkpoints/song_json/"




