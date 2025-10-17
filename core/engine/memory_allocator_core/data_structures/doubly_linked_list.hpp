#ifndef DOUBLYLINKEDLIST_HPP
#define DOUBLYLINKEDLIST_HPP 
template <class T>
class doubly_linked_list{
public:
  struct list_node{
    T node_data; 
    list_node *previous_node; 
    list_node *next_node; 
  };
  
  list_node *head_node; 

  doubly_linked_list() = default; 
  void insert_node(list_node *previous_node, list_node *new_node){
    if(previous_node == nullptr){
      if(head_node != nullptr){
        new_node->next_node = head_node; 
        new_node->next_node->previous_node = new_node; 
      }else{
        new_node->next_node = nullptr; 
      }
      head_node = new_node; 
      head_node->previous_node = nullptr; 
    }else{
      if(previous_node->next_node == nullptr){
        previous_node->next_node = new_node; 
        new_node->next_node = nullptr; 
      }else{
        new_node->next_node = previous_node->next_node;
        if(new_node->next_node != nullptr){
          new_node->next_node->previous_node = new_node;  
        }
        previous_node->next_node = new_node; 
        new_node->previous_node = previous_node; 
      }
    }
  }

  void remove_node(list_node *delete_node){
    if(delete_node->previous_node == nullptr){
      if(delete_node->next_node == nullptr){
        head_node = nullptr;
      }else{
        head_node = delete_node->next_node; 
        head_node->previous_node = nullptr;
      }
    }else{
      if(delete_node->next_node == nullptr){
        delete_node->previous_node->next_node = nullptr; 
      }else{
        delete_node->previous_node->next_node = delete_node->next_node;
        delete_node->next_node->previous_node = delete_node->previous_node;
      }
    }
  } 

}; 

#endif 
