use std::{collections::HashMap, ptr::NonNull};

use crate::cache::EntryID;

#[derive(Debug)]
struct Node {
    entry_id: EntryID,
    prev: Option<NonNull<Node>>,
    next: Option<NonNull<Node>>,
}

#[derive(Debug, Default)]
struct HashList {
    head: Option<NonNull<Node>>,
    tail: Option<NonNull<Node>>,
}

impl HashList {
    unsafe fn unlink_node(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        match node.prev {
            Some(mut prev) => unsafe { prev.as_mut().next = node.next },
            None => self.head = node.next,
        }

        match node.next {
            Some(mut next) => unsafe { next.as_mut().prev = node.prev },
            None => self.tail = node.prev,
        }

        unsafe { drop(Box::from_raw(node_ptr.as_ptr())) };
    }

    /// Pushes the node to the front (head) of the list.
    /// Must be called within the lock.
    unsafe fn push_front(&mut self, mut node_ptr: NonNull<Node>) {
        let node = unsafe { node_ptr.as_mut() };

        node.next = self.head;
        node.prev = None;

        match self.head {
            Some(mut head) => unsafe { head.as_mut().prev = Some(node_ptr) },
            None => self.tail = Some(node_ptr),
        }

        self.head = Some(node_ptr);
    }
}

impl Drop for HashList {
    fn drop(&mut self) {
        while let Some(node_ptr) = self.head {
            unsafe {
                self.unlink_node(node_ptr);
            }
        }
    }
}

#[cfg_attr(not(rust_analyzer), cfg(kani))]
mod proofs {
    use super::*;
    use std::ptr::NonNull;

    #[cfg_attr(not(rust_analyzer), kani::proof)]
    #[cfg_attr(not(rust_analyzer), kani::unwind(12))]
    fn kani_hash_list_push_front() {
        let count = kani::any::<u8>();
        kani::assume(count < 3 && count > 0);

        let mut state = HashList::default();

        for i in 0..(count as usize) {
            let entry_id: EntryID = i.into();
            let node = Node {
                entry_id,
                prev: None,
                next: None,
            };

            let node_ptr = unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(node))) };

            // state.map.insert(entry_id, node_ptr);
            unsafe {
                state.push_front(node_ptr);
            }
        }

        assert!(state.head.is_some());

        let forward_count = {
            let mut c = 0;
            let mut current = state.head;
            while let Some(ptr) = current {
                c += 1;
                current = unsafe { ptr.as_ref().next };
            }
            c
        };
        assert_eq!(forward_count, count as usize);

        let backward_count = {
            let mut c = 0;
            let mut current = state.tail;
            while let Some(ptr) = current {
                c += 1;
                current = unsafe { ptr.as_ref().prev };
            }
            c
        };
        assert_eq!(backward_count, count as usize);
    }
}
