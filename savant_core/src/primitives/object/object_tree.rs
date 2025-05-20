use super::WithId;

pub struct ObjectTree<T: WithId> {
    object: T,
    children: Vec<ObjectTree<T>>,
}

impl<T: WithId> ObjectTree<T> {
    pub fn new(object: T) -> Self {
        ObjectTree {
            object,
            children: vec![],
        }
    }
    pub fn find_object(&self, id: i64) -> Option<&T> {
        if self.object.get_id() == id {
            Some(&self.object)
        } else {
            self.children.iter().find_map(|child| child.find_object(id))
        }
    }

    pub fn find_object_mut(&mut self, id: i64) -> Option<&mut ObjectTree<T>> {
        if self.object.get_id() == id {
            Some(self)
        } else {
            self.children
                .iter_mut()
                .find_map(|child| child.find_object_mut(id))
        }
    }

    pub fn add_child(&mut self, child: ObjectTree<T>) {
        self.children.push(child);
    }

    pub fn walk_objects<F, R>(&self, mut f: F) -> anyhow::Result<()>
    where
        F: FnMut(&T, Option<&T>, Option<&R>) -> anyhow::Result<R>,
    {
        self.walk_objects_int(&mut f, None, None)?;
        Ok(())
    }

    fn walk_objects_int<F, R>(
        &self,
        f: &mut F,
        parent: Option<&T>,
        result: Option<&R>,
    ) -> anyhow::Result<R>
    where
        F: FnMut(&T, Option<&T>, Option<&R>) -> anyhow::Result<R>,
    {
        let res = f(&self.object, parent, result)?;
        for child in &self.children {
            child.walk_objects_int(f, Some(&self.object), Some(&res))?;
        }
        Ok(res)
    }

    pub fn get_object_ids(&self) -> anyhow::Result<Vec<i64>> {
        let mut ids = Vec::new();
        self.walk_objects(&mut |object: &T, _: Option<&T>, _: Option<&()>| {
            ids.push(object.get_id());
            Ok(())
        })?;
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::object::WithId;

    use super::*;

    #[test]
    fn test_new_tree() -> anyhow::Result<()> {
        struct TestObject {
            id: i64,
        }

        impl WithId for TestObject {
            fn get_id(&self) -> i64 {
                self.id
            }

            fn set_id(&mut self, id: i64) {
                self.id = id;
            }
        }

        let mut tree = ObjectTree::new(TestObject { id: 1 });
        tree.add_child(ObjectTree::new(TestObject { id: 2 }));
        tree.add_child(ObjectTree::new(TestObject { id: 3 }));
        let object_3 = tree.find_object_mut(3).unwrap();
        object_3.add_child(ObjectTree::new(TestObject { id: 4 }));
        object_3.add_child(ObjectTree::new(TestObject { id: 5 }));
        let object_5 = tree.find_object_mut(5).unwrap();
        object_5.add_child(ObjectTree::new(TestObject { id: 6 }));
        assert_eq!(tree.find_object(1).unwrap().id, 1);
        assert_eq!(tree.find_object(2).unwrap().id, 2);
        assert_eq!(tree.find_object(3).unwrap().id, 3);
        assert_eq!(tree.find_object(4).unwrap().id, 4);
        assert_eq!(tree.find_object(5).unwrap().id, 5);
        assert_eq!(tree.find_object(6).unwrap().id, 6);

        let mut visited = Vec::new();
        tree.walk_objects(&mut |object: &TestObject,
                                parent: Option<&TestObject>,
                                result: Option<&i64>| {
            let parent_id = parent.map(|p| p.id).unwrap_or(-1);
            visited.push((object.id, parent_id, *result.unwrap_or(&-1)));
            Ok(object.id)
        })?;
        assert_eq!(
            visited,
            vec![
                (1, -1, -1),
                (2, 1, 1),
                (3, 1, 1),
                (4, 3, 3),
                (5, 3, 3),
                (6, 5, 5)
            ]
        );
        Ok(())
    }
}
