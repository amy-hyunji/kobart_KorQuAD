import torch

from tqdm import tqdm

debug=False

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        if 'answer_end' in answer.keys():
            continue
        else:
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two - fix this 
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx 
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx-1
                answer['answer_end'] = end_idx-1
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx-2
                answer['answer_end'] = end_idx-2 
            assert (answer['answer_start'] <= len(context))
            assert (answer['answer_end'] <= len(context))
    return

def text_processing(data, args, tokenizer):
    input_ids = []
    attn_mask_ids = []
    label_start = [] # char start idx in context 
    label_end = []   # char end idx in context
    max_len = args.max_len
    data_num = len(data['context'])
   
    # contain context with no answer
    n_input_ids = []
    n_attn_mask_ids = []
    n_label_start = [] 
    n_label_end = []  

    cleanse = args.cleanse

    for idx in tqdm(range(data_num)):

        context = data['context'][idx]
        question = data['question'][idx] 
        answer = data['answer'][idx]
        start_idx = answer['answer_start'] # start idx of the char in context
        end_idx = answer['answer_end']     # end idx+1 of the char in context
        answer_text = answer['text']

        prev_sen = context[:start_idx]
        assert (answer_text == context[start_idx:end_idx])

        prev_ids = tokenizer.encode(prev_sen, add_special_tokens=False)
        context_ids = tokenizer.encode(context, add_special_tokens=False) 
        # get start_token 
        # case 0 - when there is no answer:
        if answer_text == None:
            start_token = 0
        # case 1 - when the first elem of answer text is connected to last elem of prev_ids
        elif answer_text[0] in tokenizer.convert_ids_to_tokens(context_ids[len(prev_ids)-1]):
            start_token = len(prev_ids)-1 
        # case 2 - when they are not connected 
        elif answer_text[0] in tokenizer.convert_ids_to_tokens(context_ids[len(prev_ids)]):
            start_token = len(prev_ids)
        elif answer_text[0] in tokenizer.convert_ids_to_tokens(context_ids[len(prev_ids)+1]):
            start_token = len(prev_ids)+1
        else:
            print("ERROR!!! in finding start_token")
            continue
            """
            print(f"context: {context}")
            print(tokenizer.convert_ids_to_tokens(context_ids))
            print(f"answer_text: {answer_text}")
            print(f"answer_text[0]: {answer_text[0]}")
            print(f"len(prev_ids): {len(prev_ids)}")
            print(f"context_ids[len(prev_ids)]: {tokenizer.convert_ids_to_tokens(context_ids[len(prev_ids)])}")
            assert (False)
            """

        # starting from the start_token, go through context_ids till all the answers are out
        possible_ans = ""
        m = 0
        end_token = 0
        error = False
        while not answer_text in possible_ans:
            if (answer_text == None): break
            if (start_token+m == len(context_ids)): 
                error = True
                #print(f"Error occured in context: {context}, answer: {answer}")
                break
            possible_ans += tokenizer.convert_ids_to_tokens(context_ids[start_token+m]).replace("▁", " ")
            m+=1
            #print(f"possible_ans: {possible_ans}")
            end_token = start_token + m
        if error: continue 
        assert (answer_text in possible_ans)

        q = tokenizer.encode(question, add_special_tokens=False)
        if not answer_text == None: 
            a = tokenizer.encode(answer_text, add_special_tokens=False)
        else:
            a = None
        c = context_ids

        spair_len = max_len - 4 - len(q) # length left for context 
        assert (spair_len > 0)

        chunk = []
        if (len(c) <= spair_len):
            # no problem. just make the format 
            chunk.append(c)

        else:
            # need to divide context into chunk
            iter_num = len(c)//spair_len + 1 if len(c)%spair_len != 0 else len(c)//spair_len
            for _iter in range(iter_num):
                chunk.append(c[_iter*spair_len:(_iter+1)*spair_len])


        stack_idx = 0
        for i, _chunk in enumerate(chunk):
            if debug: print(f"chunk #: {len(chunk)}, stack_idx: {stack_idx}")
            # end = start = 0 if not exist in _chunk

            _chunk_len = len(_chunk) + len(q) + 4
            _input_ids = [tokenizer.cls_token_id] + _chunk + [tokenizer.sep_token_id]*2 + q + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_len-_chunk_len)  
            assert(len(_input_ids) == max_len)
            _attn_mask_ids = [1]*_chunk_len + [0]*(max_len-_chunk_len)

            _start_idx = start_token - stack_idx      # idx in current chunk
            _end_idx = end_token - stack_idx          # idx in current chunk
    
            # no answer 
            if a is None:
                _label_start = 0
                _label_end = 0

            # already extracted answer from previous chunk
            if (_start_idx < 0 and _end_idx <= 0):
                if debug: print("case 1: answers already from previous chunk")
                _label_start = 0
                _label_end = 0

            # second chunk with partial answer -> 
            elif (_start_idx < 0 and _end_idx > 0):
                # answer ends in this chunk
                if _end_idx <= len(_chunk):
                    _label_start = 1
                    _label_end = _end_idx + 1
                # answer connected to next chunk
                else:
                    _label_start = 1
                    _label_end = len(_chunk) + 1
                if debug: print(f"case 2: partial answer second chunk, ans: {tokenizer.convert_ids_to_tokens(_input_ids[_label_start:_label_end])}")

            elif (_start_idx >= 0 and _end_idx < 0):
                print("Shouldn't be here!\ncontext: {context}\nchunk: {_chunk}\nanswer: {answer}")
                sys.exit(-1)

            else:
                # answer in later chunk 
                if len(_chunk) <= _start_idx:
                    if debug: print(f"case 3: Answer in later chunk")
                    _label_start = 0
                    _label_end = 0
                elif _start_idx < len(_chunk):
                    # first chunk with partial answer
                    if len(_chunk) <= _end_idx:
                        _label_start = _start_idx + 1
                        _label_end = len(_chunk) + 1
                        if debug: print(f"case 4: partial answer first chunk, ans: {tokenizer.convert_ids_to_tokens(_input_ids[_label_start:_label_end])}")
                    # all in current chunk
                    else:
                        _label_start = _start_idx + 1
                        _label_end = _end_idx + 1
                        if debug: print(f"case 5: all in here!, ans: {tokenizer.convert_ids_to_tokens(_input_ids[_label_start:_label_end])}")

            if debug: print("answer!!: ",tokenizer.convert_ids_to_tokens(_chunk[_start_idx:_end_idx]))
            stack_idx += len(_chunk)


            ### append all elements
            if cleanse and (_label_start == _label_end == 0):
                n_input_ids.append(torch.tensor(_input_ids, dtype=torch.long) )
                n_attn_mask_ids.append(torch.tensor(_attn_mask_ids, dtype=torch.long))
                assert (_label_start == 0 and _label_end == 0)
                n_label_start.append(torch.tensor(_label_start, dtype=torch.long))
                n_label_end.append(torch.tensor(_label_end, dtype=torch.long))
                continue

            input_ids.append(torch.tensor(_input_ids, dtype=torch.long) )
            attn_mask_ids.append(torch.tensor(_attn_mask_ids, dtype=torch.long))
            assert (_label_start >= 0 and _label_end >= 0)
            assert (_label_end <= len(_chunk)+1)
            # +1 for cls token in placement 0
            label_start.append(torch.tensor(_label_start, dtype=torch.long))
            label_end.append(torch.tensor(_label_end, dtype=torch.long))

            """
            print(f"context: {context}")
            print(f"question: {question}")
            print(f"answer: {answer}")
            print(f"chunk: {tokenizer.convert_ids_to_tokens(_chunk)}")
            print(f"possible_answer: {tokenizer.convert_ids_to_tokens(_input_ids[_label_start:_label_end])}")
            print(" ")
            """

    assert(len(input_ids) == len(attn_mask_ids) == len(label_start) == len(label_end))
    assert(len(n_input_ids) == len(n_attn_mask_ids) == len(n_label_start) == len(n_label_end))

    # add save number of contexts to `input_ids` from `n_input_ids`
    if cleanse:
        _num = len(input_ids)
        input_ids += n_input_ids[:_num]
        attn_mask_ids += n_attn_mask_ids[:_num]
        label_start += n_label_start[:_num]
        label_end += n_label_end[:_num]
        print(f"*** CLEANSE! # of context was {_num}.\n***Add empty context. Now {len(input_ids)}")

    return {'input_ids': input_ids, 'attn_mask_ids': attn_mask_ids, 'label_start': label_start, 'label_end': label_end} 
