## Concept of BPE 
- BPE is a data compression and tokenization algorithm that replaces the most frequent pair of consecutive bytes or characters with new bytes or characters not present in the data. In the context of NLP, BPE creates sub-word tokens by merging frequent character sequences. 
## Phases of the algorithm 
- Training: Identifying and merging the most frequent adjacent token pairs. 
- Encoding: Applying learned merge rules to tokenize new text. 
## Data structures and type definitions 
```c++
using g_token_id = uint32_t; 
using g_token_pair = std::pair<g_token_id, g_token_id>; 
```
- `g_token_id`: A 32-bit unsigned integer used to identify tokens. 
- `g_token_pair`: A pair of token Id's that represent adjacent tokens that might be merged. 
```c++
struct token_pair_hash{
	size_t operator() (const g_token_pair &token_pair) const{ 
		return std::hash<g_token_id>()(pair.first) ^ 
		(std::hash<g_token_id>()(pair.second) << 1); 
	}
}
```
- The above hash function takes a token pair and produces a unique hash value, enabling token pairs to be keys placed in unordered maps. It makes use of the XOR (^) operator through bit-shifting to combine hashes from both tokens, thus reducing the probability of collisions. 
## BPE Tokenizer class components
### Core data members 
```c++
std::unordered_map<std::string, g_token_id> token_to_id; 
std::unordered_map<g_token_id, std::string> id_to_token; 
g_token_id = 0; 
m_next_id  = 0; 
const g_token_id UNK_TOKEN = UINT32_MAX; 
```
- `token_to_id`: Maps token strings to their unique ID's through **Forward mapping**. 
- `id_to_token`: Maps token ID's back to their string representations through **Reverse mapping**.
- `m_next_id`: Tracks the next available ID for any new tokens. 
- `UNK_TOKEN`: Is a special ID for unknown tokens, set to a maximum possible value. 
```c++
struct merging_rule{
	g_token_id first; 
	g_token_id second; 
	g_token_id merged; 
}
```
- `merging_rule`: Records one BPE merge operation with the IDs of tokens being merged and the resulting new token ID. 
- `merge_history`: Will store the entire history of the merge operations in the order they were learned. 
### Constructor
```c++
bpe_tokenizer(){
	for(int i = 0; i < 256; ++i){
		std::string s(1, static_cast<char>(i)); 
		add_token(s); 
	}
}
```
- The constructor will initialize the tokenizer with all 256 possible byte values as individual tokens. This will create the base vocabulary from which all possible merges will be derived. 
### Token management
 ```c++
 g_token_id add_token(const std::string &token){ 
	 if(token_to_id.find(token) == token_to_id.end()){
		 token_to_id[token] = m_next_id; 
		 id_to_token[m_next_id] = token; 
		 return m_next_id++; 
	 }
	 return token_to_id[token]; 
 }
```
- The above `add_token()` function adds a new token to the vocabulary if it doesn't currently exist. It does so by updating both mapping dictionaries, `token_to_id` and `id_to_token`. It will then return the token's ID (either the new ID or an existing one if found) and will then increment `m_next_id` for the next token. 
```c++
g_token_id get_id(const std::string &token) const{
	try{
		return token_to_id.at(token); 
	}catch(const std::out_of_range&){
		return UNK_TOKEN;
	}
}
```
- `get_id()` will safely find and return the ID for a given token, if the token is not found in the vocabulary, it will simply return the `UNK_TOKEN` token type. 
### Training algorithm 
```c++
void train(const std::string &text, size_t merges, double regularization = 0.0){
	if( text.empty() || merges == 0){
		return; 
	}
	std::vector<g_token_id> tokens; 
	for(auto &c : text){
		tokens.push_back(token_to_id[std::string(1, c)]);
	}
	//More next ...
}
```
- The training pass starts off by converting the input text into individual character tokens, it will return early if the text is empty or if no merges are requested. 
- It will then create a new vector of token IDs from the individual characters. 
```c++
//From above ...
for(size_t merge = 0; merge < merges; ++merges){
	std::unordered_map<g_token_pair, int, token_pair_hash> token_pair_counts; 
	for(size_t i = 0; i < tokens.size() - 1; ++i){
		g_token_pair pair = std::make_pair(tokens[i], tokens[i + 1]); 
		token_pair_counts[pair]++; 
	}
	if(token_pair_counts.empty()){
		break; 
	}
	//More next ...
}
```
- For each merge iteration, the algorithm will create a map to count occurrences of each adjacent token pair, it then iterates through the tokens whilst counting each pair and breaks if no pairs remain, meaning all tokens have been merged. 
```c++
//From above ... 
using pair_count = std::pair<double, g_token_pair>; 
std::priority_queue<pair_count> token_queue;

for(const auto &[pair, count] : token_pair_counts){
	double score =  count; 
	if(regularization > 0.0){
		const std::string &first  = id_to_token[pair.first]; 
		const std::string &second = id_to_token[pair.second];
		score -= regularization  * (first.length() + second.length());  
	}
	token_queue.emplace(score,pair); 
	//More next ...
}
```
- Next it creates a priority queue to find the highest-frequency pair, it will then calculate a score for each pair based on it's frequency (higher is better) and the optional regularization to prefer shorter tokens. This works by subtracting a penalty proportional to the token's length. The priority queue will then automatically sort pairs by their scores. 
```c++
//From above ...
auto [max_score, best_pair] = token_queue.top(); 
auto [a ,b] = best_pair; 

std::string merged = id_to_token[a] + id_to_token[b]; 
g_token_id new_id = add_token(merged); 

merge_history.push_back({a, b, new_id});
```
- Next it extracts the highest scoring pair from the queue and creates a new token by concatenating the strings of the pair. It then adds the new token to the vocabulary and records the merge operation to the `merge_history`. 
```c++
//From above ...
std::vector<g_token_id> new_tokens; 
size_t i = 0; 
while(i < tokens.size()){
	if(i < tokens.size() - 1 && tokens[i] == a && tokens[i + 1] == b){
		new_tokens.push_back(new_id); 
		i += 2;
	}else{
		new_tokens.push_back(tokens[i]); 
		i++;
	}
}
tokens = new_tokens; 
```
- Finally, it applies the new merge rule to the current token list by, creating a new token vector and scanning through the tokens. When the pair `(a,b)` is found, it's replaced with the merged token, otherwise it keeps the original token. Finally the the token list is updated for the next iteration of the training cycle. 
### Encoding tokens
```c++
std::vector<g_token_id> encode(const std::string &text) const{
	if(text.empty()){
		return {}; 
	}
	std::vector<g_token_id> tokens; 
	for(auto &c : text){
		std::string s(1, c); 
		try{
			tokens.push_back(token_to_id.at(s));
		}catch(const std::out_of_range&){
			tokens.push_back(UNK_TOKEN);
		}
	}
	//more next ... 
}
```
- The encoding process starts by breaking the text into individual characters, each character is then converted to it's corresponding token ID. Unknown characters get assigned an `UNK_TOKEN`ID type, it returns an empty vector for empty inputs. 
```c++
//from above ... 
for(const auto &rule : merge_history){
	std::vector<g_token_id> new_tokens; 
	size_t i = 0; 
	while(i < tokens.size()){
		if(i < tokens.size() - 1 && tokens[i] == rule.first &&
		  tokens[i + 1] == rule.second){
			  new_tokens.push_back(rule.merged);
			  i += 2; 
		}else{
			  new_tokens.push_back(tokens[i]); 
			  i++; 
		}
	}
	tokens = std::move(new_tokens); 
}
```
- Next it applies each merge rule from the training phase in order, for each rule it will: create a new token vector, it'll scan through the current tokens and when a pair matches the current merge rule, it will replace the current token with the merged token, otherwise, it will keep the original token. 
### Decoding IDs 
```c++
std::string decode(const std::vector<g_token_id> &tokens) const{
	std::string result; 
	for(auto &id : tokens){
		try{
			result += id_to_token.at(id);
		}catch(const std::out_of_range&){
			result += "<UNK>"; 
		}
	}
	return result; 
}
``` 
- The decoding algorithm essentially converts a sequence of token IDs back into text. To achieve this, it looks up each token ID in the `id_to_token` map and concatenates the string representations. It handles unknown token IDs by inserting the unknown token ID `<UNK>`. 
### Model Saving 
```c++
bool save_model(const std::string &vocab_path, const std::string &merge_path) const{
	try{
		std::ofstream vocab_file(vocab_path); 
		if(!vocab_file.is_open()){
			return false; 
		}
		for(const auto &[token, id] : token_to_id){
			vocab_file << token << "\t" << id << "\n"; 
		}
		vocab_file.close(); 
		//more next ... 
}
```
- The `save_model` process starts by saving the entire token vocabulary into a file, each line of the file contains a token and it's corresponding ID. It uses a tab space as a separator and returns false if the file can't be opened. 
```c++
//from above ... 
	std::ofstream merge_file(merge_path); 
	if(!merge_file.is_open()){
		return false; 
	}
	for(const auto &rule : merge_history){
		merge_file << rule.first << "\t" << rule.second 
		           << "\t" << rule.merged << "\n";
	}
	merge_file.close(); 
	return true;
}catch(const std::exception&){
	return false; 
}
```
- Next it saves the merge rules into a separate file where each line contains the first token ID, second token ID and the merged token ID using a tab space as a separator. It also returns false if the file can't be opened. 
#### Model loading
```c++
bool load_model(const std::string &vocab_path, const std::string &merge_path){
	try{
	token_to_id.clear(); 
	id_to_token.clear(); 
	merge_history.clear(); 
	//more next ... 
}
```
- The `load_model` function starts out by clearing existing data for a fresh start 
```c++
	std::ifstream vocab_file(vocab_path); 
	if(!vocab_file.is_open()){
		return false; 
	}
	std::string line; 
	std::string token; 
	g_token_id id; 
	while(std::getline(vocab_file, line)){
		std::istringstream iss(line); 
		if(iss >> token >> id){
			token_to_id[token] = id; 
			id_to_token[id]    = token; 
			if(id >= m_next_id){
				m_next_id = id + 1; 
			}
		}
	 }
```
- Next it reads the vocabulary from the vocabulary file, each line is parsed to extract the token and ID, it updates both token maps and also updates `m_next_id` to be greater than any seen ID. 
```c++
	std::ifstream merge_file(merge_path); 
	if(!merge_file.is_open()){
		return false; 
	}
	g_token_id first; 
	g_token_id second; 
	g_token_id merged; 
	std::string merge_line;
	while(std::getline(merge_file, merge_line)){
		std::istringstream iss(merge_line)
		if(iss >> first >> second >> merged){
			merge_history.push_back({first, second, merged}); 
		}
	}
```
- Finally, it reads the merge rules from the merge file, it parses each line to extract the first, second and merged token IDs and rebuilds the entire merge history in order. 
## How BPE works
- BPE essentially works by identifying and merging frequent token pairs, this creates a vocabulary that efficiently represents text by: 
	- Starting with character level tokens 
	- Creating new tokens for common character sequences 
	- Building larger tokens for frequently co-occurring sequences 
-  This results in a vocabulary that adapts to the texts' patterns and allows efficient representation with sub-word units.  
### How merges affect encoding 
- **Fewer merges:** Encoding stays close to character level with a small number of common pairs merged.
- **More merges:** Encoding begins to capture common sequences, words and word parts. 
- **Many merges:** Encoding can represent entire common words or phrases as single tokens. 
- A key insight is that the tokenization process adapts to the specific data patters rather than using predefined word boundaries. 
### Order of operations 
- During the training, the most frequent merges are found and applied at each step 
- Each merge is recorded in sequence 
- During the encoding process, these merges are applied in the same exact order they were found, this ensures consistent tokenization 
- If the merges were applied in a different order, earlier merges would affect the available pairs for later merges, thus skewing our results 
## Performance 
### Time complexity 
#### Training 
- $O(n*m)$ where $n$ is the length of the input text and $m$ is the total number of merges 
- Each merge requires scanning all tokens and counting pairs $O(n)$
- Finding the most frequent pairs takes $O(p \log p)$ where $p$ is the number of unique pairs 
- All of this is done $m$ times for $m$ number of merges 
#### Encoding 
- $O(n*m)$ where $n$ is the length of the input text and $m$ is the total number of merges 
- Each merge rule requires one pass through the current tokens 
- This is done $m$ times for $m$ number of merges 
### Space complexity 
- **Vocabulary:** $O(v)$ where $v$ is the vocabulary size, initially 256 chars and merges 
- **Merge history:** $O(m)$ where $m$ is the total number of merges 
- **Token storage:** $O(n)$ where $n$ is the length of the input text 