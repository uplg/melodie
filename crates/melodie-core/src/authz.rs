use crate::ids::{SongId, UserId};
use crate::model::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Read,
    Write,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    Song { owner_id: UserId, song_id: SongId },
}

pub fn can(actor_role: Role, actor_id: UserId, action: Action, resource: Resource) -> bool {
    if actor_role == Role::Admin {
        return true;
    }
    match (action, resource) {
        (Action::Read | Action::Write | Action::Delete, Resource::Song { owner_id, .. }) => {
            owner_id == actor_id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_can_anything() {
        let admin = UserId::new();
        let owner = UserId::new();
        let song = SongId::new();
        assert!(can(
            Role::Admin,
            admin,
            Action::Delete,
            Resource::Song {
                owner_id: owner,
                song_id: song,
            },
        ));
    }

    #[test]
    fn member_only_owns() {
        let me = UserId::new();
        let other = UserId::new();
        let song = SongId::new();
        assert!(can(
            Role::Member,
            me,
            Action::Read,
            Resource::Song {
                owner_id: me,
                song_id: song,
            },
        ));
        assert!(!can(
            Role::Member,
            me,
            Action::Read,
            Resource::Song {
                owner_id: other,
                song_id: song,
            },
        ));
    }
}
