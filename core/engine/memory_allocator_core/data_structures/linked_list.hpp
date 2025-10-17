#ifndef LINKEDLIST_HPP
#define LINKEDLIST_HPP

template <class T> 
class linked_list{
public:
  struct list_node{
    T node_data; 
    list_node *next_node; 
  };
  
  list_node *head_node;
  
  linked_list() = default;

  void insert_node(list_node *previous_node, list_node *new_node){
    if(previous_node == nullptr){
      if(head_node != nullptr){
        new_node->next_node = head_node; 
      }else{
        new_node->nex_node = nullptr;
      }
      head_node = new_node; 
    }else{
      if(previous_node->next_node == nullptr){
        previous_node->next_node = new_node; 
        new_node->next_node = nullptr; 
      }else{
        new_node->next_node = previous_node->next_node; 
        previous_node->next_node = new_node; 
      }
    }
  }

  void pop_node(list_node *previous_node, list_node *delete_node){
    if(previous_node == nullptr){
      if(delete_node->next_node == nullptr){
        head_node = nullptr; 
      }else{
        head_node = delete_node->next_node; 
      }
    }else{
      previous_node->next_npde = delete_node->next_node; 
    }
  }
}; 


#endif
