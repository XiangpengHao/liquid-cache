create table PostHistoryTypes (
    Id   smallint    not null,
    Name varchar(50) not null,
    primary key (Id)
);
create table LinkTypes (
    Id   smallint    not null,
    Name varchar(50) not null,
    primary key (Id)
);
create table PostTypes (
    Id   smallint    not null,
    Name varchar(50) not null,
    primary key (Id)
);
create table CloseReasonTypes (
    Id   smallint    not null,
    Name varchar(50) not null,
    primary key (Id)
);
create table VoteTypes (
    Id   smallint    not null,
    Name varchar(50) not null,
    primary key (Id)
);

create table Users (
    Id              int       not null primary key,
    Reputation      int       not null,
    CreationDate    timestamp not null,
    DisplayName     varchar(40),
    LastAccessDate  timestamp not null, /* The time when the user last loaded a page; updated every 30 min at most */
    WebsiteUrl      varchar(200),
    Location        varchar(300),
    AboutMe         text,
    Views           int, /* Number of times the profile is viewed */
    UpVotes         int, /* How many upvotes the user has cast */
    DownVotes       int, /* How many downvotes the user has cast */
    ProfileImageUrl varchar(200),
    AccountId       int /* User's Stack Exchange Network profile ID */
);

create table Badges (
    Id       int         not null primary key,
    UserId   int         not null,
    Name     varchar(50) not null, /* Name of the badge */
    Date     timestamp   not null, /* Date the badge was awarded, e.g. 2008-09-15T08:55:03.923 */
    Class    smallint    not null, /* 1 (Gold), 2 (Silver), or 3 (Bronze) */
    TagBased bool        not null /* True if badge is for a tag, otherwise it is a named badge */
);

create table Posts (
    Id                    int not null primary key,
    PostTypeId            smallint references PostTypes (Id), /* (listed in the PostTypes table)
    - 1 = Question
    - 2 = Answer
    - 3 = Wiki
    - 4 = TagWikiExcerpt
    - 5 = TagWiki
    - 6 = ModeratorNomination
    - 7 = WikiPlaceholder (Appears to include auxiliary site content like the help center introduction, election description, and the tour page's introduction, ask, and don't ask sections)
    - 8 = PrivilegeWiki*/
    AcceptedAnswerId      int, /* only present if PostTypeId = 1 */
    ParentId              int, /* only present if PostTypeId = 2 */
    CreationDate          timestamp,
    Score                 int, /* generally non-zero for only Questions, Answers, and Moderator Nominations */
    ViewCount             int,
    Body                  text, /* as rendered HTML, not Markdown */
    OwnerUserId           int references Users (Id), /* only present if user has not been deleted; always -1 for tag wiki entries, i.e. the community user owns them */
    OwnerDisplayName      varchar(40),
    LastEditorUserId      int references Users (Id),
    LastEditorDisplayName varchar(40),
    LastEditDate          timestamp, /* e.g. 2009-03-05T22:28:34.823 - the date and time of the most recent edit to the post */
    LastActivityDate      timestamp, /* e.g. 2009-03-11T12:51:01.480 - datetime of the post's most recent activity */
    Title                 varchar(300), /* question title (PostTypeId = 1), or on Stack Overflow, the tag name for some tag wikis and excerpts (PostTypeId = 4/5) */
    Tags                  varchar(4000), /* question tags (PostTypeId = 1), or on Stack Overflow, the subject tag of some tag wikis and excerpts (PostTypeId = 4/5) */
    AnswerCount           int, /* the number of undeleted answers (only present if PostTypeId = 1) */
    CommentCount          int,
    FavoriteCount         int,
    ClosedDate            timestamp, /* present only if the post is closed */
    CommunityOwnedDate    timestamp, /* present only if post is community wiki'd */
    ContentLicense        varchar(30)
);

create table Comments (
    Id              int           not null primary key,
    PostId          int           not null references Posts (Id),
    Score           int,
    Text            varchar(2000) not null, /* Comment body */
    CreationDate    timestamp     not null,
    UserDisplayName varchar(40),
    UserId          int references Users (Id), /* optional, absent if user has been deleted */
    ContentLicense  varchar(30)
);

create table PostHistory (
    Id                int not null primary key,
    PostHistoryTypeId smallint references PostHistoryTypes (Id), /* (listed in the PostHistoryTypes table)
    - 1 = Initial Title (initial title (questions only))
    - 2 = Initial Body (initial post raw body text)
    - 3 = Initial Tags (initial list of tags (questions only)
    - 4 = Edit Title (modified title (questions only))
    - 5 = Edit Body (modified post body (raw markdown))
    - 6 = Edit Tags (modified list of tags (questions only))
    - 7 = Rollback Title (reverted title (questions only))
    - 8 = Rollback Body (reverted body (raw markdown))
    - 9 = Rollback Tags (reverted list of tags (questions only))
    - 10 = Post Closed (post voted to be closed)
    - 11 = Post Reopened (post voted to be reopened)
    - 12 = Post Deleted (post voted to be removed)
    - 13 = Post Undeleted (post voted to be restored)
    - 14 = Post Locked (post locked by moderator)
    - 15 = Post Unlocked (post unlocked by moderator)
    - 16 = Community Owned (post now community owned)
    - 17 = Post Migrated (post migrated - now replaced by 35/36 (away/here))
    - 18 = Question Merged (question merged with deleted question)
    - 19 = Question Protected (question was protected by a moderator)
    - 20 = Question Unprotected (question was unprotected by a moderator)
    - 22 = Question Unmerged (answers/votes restored to previously merged question)
    - 24 = Suggested Edit Applied
    - 25 = Post Tweeted
    - 31 = Discussion moved to chat
    - 33 = Post Notice Added (comment contains foreign key to PostNotices)
    - 34 = Post Notice Removed (comment contains foreign key to PostNotices)
    - 35 = Post Migrated Away (replaces id 17)
    - 36 = Post Migrated Here (replaces id 17)
    - 37 = Post Merge Source
    - 38 = Post Merge Destination
    - 50 = CommunityBump (bumped by community user)
    - 52 = SelectedHotQuestion (question became hot network question)
    - 53 = RemovedHotQuestion (question removed from hot network)
    - 66 = CreatedFromWizard */
    PostId            int references Posts (Id),
    RevisionGUID      varchar(36), /* At times more than one type of history record can be recorded by a single action. All of these will be grouped using the same RevisionGUID */
    CreationDate      timestamp,
    UserId            int references Users (Id),
    UserDisplayName   varchar(40), /* populated if a user has been removed and no longer referenced by user id */
    Comment           varchar(800), /* This field will contain the comment made by the user who edited a post
    - If PostHistoryTypeId = 10, this field contains the CloseReasonId of the close reason (listed in CloseReasonTypes):
        - Old close reasons:
            - 1 = Exact Duplicate
            - 2 = Off-topic
            - 3 = Subjective and argumentative
            - 4 = Not a real question
            - 7 = Too localized
            - 10 = General reference
            - 20 = Noise or pointless (Meta sites only)
        - Current close reasons:
            - 101 = Duplicate
            - 102 = Off-topic
            - 103 = Needs details or clarity
            - 104 = Needs more focus
            - 105 = Opinion-based
    - If PostHistoryTypeId in (33,34) this field contains the PostNoticeId of the PostNotice */
    Text              text, /*  A raw version of the new value for a given revision
    - If PostHistoryTypeId in (10,11,12,13,14,15,19,20,35) this column will contain a JSON encoded string with all users who have voted for the PostHistoryTypeId
    - If it is a duplicate close vote, the JSON string will contain an array of original questions as OriginalQuestionIds
    - If PostHistoryTypeId = 17 this column will contain migration details of either from <url> or to <url> */
    ContentLicense    varchar(30)
);

create table PostLinks (
    Id            bigint    not null primary key,
    CreationDate  timestamp not null, /* when the link was created */
    PostId        int       not null references Posts (Id), /* id of source post */
    RelatedPostId int       not null references Posts (Id), /* id of target/related post */
    LinkTypeId    smallint  not null references LinkTypes (Id) /* (listed in the LinkTypes table)
    - 1 = Linked (PostId contains a link to RelatedPostId)
    - 3 = Duplicate (PostId is a duplicate of RelatedPostId) */
);

create table Tags (
    Id              int not null primary key,
    TagName         varchar(35),
    Count           int not null,
    ExcerptPostId   int references Posts (Id), /* Id of Post that holds the excerpt text of the tag */
    WikiPostId      int references Posts (Id), /* Id of Post that holds the wiki text of the tag */
    IsModeratorOnly bool,
    IsRequired      bool
);

create table Votes (
    Id           int      not null primary key,
    PostId       int      not null, /* references Posts (Id), (not enforced in the data) */
    VoteTypeId   smallint not null references VoteTypes (Id), /*  (listed in the VoteTypes table)
    - 1 = AcceptedByOriginator
    - 2 = UpMod (AKA upvote)
    - 3 = DownMod (AKA downvote)
    - 4 = Offensive
    - 5 = Favorite (AKA bookmark; UserId will also be populated) feature removed after October 2022 / replaced by Saves
    - 6 = Close (effective 2013-06-25: Close votes are only stored in table: PostHistory)
    - 7 = Reopen
    - 8 = BountyStart (UserId and BountyAmount will also be populated)
    - 9 = BountyClose (BountyAmount will also be populated)
    - 10 = Deletion
    - 11 = Undeletion
    - 12 = Spam
    - 14 = NominateModerator
    - 15 = ModeratorReview (i.e., a moderator looking at a flagged post)
    - 16 = ApproveEditSuggestion */
    UserId       int references Users (Id),
    CreationDate timestamp,
    BountyAmount int
);
