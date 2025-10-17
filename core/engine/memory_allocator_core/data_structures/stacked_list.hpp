#ifndef STACKEDLIST_HPP
#define STACKEDLIST_HPP 

template <class T> 
class stacked_list{

public: 
  struct list_node{
    T node_data; 
    list_node *next_node; 
  }; 
  
  list_node *head_node; 
  
  stacked_list() = default; 
  
  stacked_list(stacked_list &list_in) = delete; 
  
  void push_node(list_node *new_node){
    new_node->next_node = head_node; 
    head_node = new_node; 
  }

  list_node *pop_node(){
    list_node *top_node = head_node; 
    head_node = head_node->nex_node; 
    return top_node; 
  } 
};

#endif 
