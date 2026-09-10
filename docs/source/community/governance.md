(target-governance)=
# Governance

`movement` is led by the [Neuroinformatics Unit](https://neuroinformatics.dev/) (NIU)
at the [Sainsbury Wellcome Centre](https://www.sainsburywellcome.org/web/),
but it's very much a community effort, and we're grateful to everyone who helps move
it forward. The roles below describe the different ways people take part and the
repository access that comes with each. To see who currently fills them, head over to
the [People](target-people) page.

## Roles

### Contributors

If you've opened a pull request, filed an issue, or joined a discussion,
you're a contributor. No special access is required, and every bit of help counts.
Our [contributing guide](target-contributing) is the place to start.

Everyone who pitches in—whether through code, documentation, sample data, or discussions
that help shape the project's direction—is credited on the [People](target-people) page.
Once your first pull request is merged, a regular job adds you to that page,
and you'll also get a mention in the [release notes](movement-github:releases)
for that version of `movement`.

We strongly believe that volunteer contributions directly help us achieve our [mission](target-mission).
The more people get involved, the more robust and versatile `movement` becomes.
Contributors bring fresh perspectives, ensuring that the software serves a broader and more diverse community of users.
Beyond technical improvements, contributions help us build trust, foster shared ownership,
make the project more sustainable, and strengthen the wider ecosystem of
open-source tools for behavioural analysis.

### Trusted contributors

**Trusted contributors** are community members who have shown consistent involvement
in the project and have demonstrated that they can help maintain `movement` with care.

**Trusted contributors** are welcome to help with a range of tasks, including:

- keeping the issue tracker and pull requests tidy and welcoming;
- participating in discussions on [Zulip](movement-zulip:) and attending community calls;
- reviewing pull requests—especially in parts of the codebase they are familiar with.

On GitHub, this role comes with *Write* access. We trust members to use that judiciously:
merging straightforward changes, particularly in areas they know well,
while seeking input from **core developers** for larger or more contentious decisions.

There's no expected time commitment and no quota to meet.
Some people take on this role for a fixed period—for instance while doing an internship with us,
or while working on `movement` as part of a collaboration.

For some contributors, this role is a stepping stone towards becoming a **core developer**.
For others, it is simply a rewarding place to make a lasting contribution.
Both are entirely fine—the role is a valued destination in its own right,
and there is no expectation that you continue beyond it.

:::{note}
This role is inspired by [napari's Triage team](napari:developers/coredev/triage.html),
whose thoughtful approach to community maintenance we've adapted to fit `movement`.
:::

### Core developers

**Core developers** are the stewards of `movement`, responsible for the long-term health of the project.
They hold *Maintain* access on GitHub, which is why you'll sometimes see them called maintainers.

Beyond everything a **trusted contributor** can do, **core developers** perform the following tasks:

- manage the repository itself, including merging pull requests and making releases;
- manage the community by moderating conversations on GitHub and Zulip, organising community calls, and posting on social media;
- make decisions about the project's scope and roadmap, and prioritise features and bug fixes;
- publicly represent the project in talks, workshops, and other events.

Administrative access to the GitHub repository, the `movement` package on PyPI,
and our [Zulip chat](movement-zulip:) is reserved for **core developers** only.
Typically, administrative rights will be held by the head of the NIU
and at least one other **core developer**, to ensure continuity in case of absence.

## Decision-making

Most day-to-day decisions are made by
lazy consensus[^consensus] among the **core developers**.
Approval from one **core developer** is enough to merge most pull requests.
We only escalate to the full team for matters that affect the project's
core architecture, scope, roadmap, or governance.

For these more consequential decisions, we discuss publicly on GitHub, Zulip
or in community calls, and consult **trusted contributors** and the wider community for input.
We aim to reach a consensus[^consensus] among **core developers**. In the rare cases we can't,
the head of the NIU has the final say and responsibility to decide for the project.

## Joining and stepping down

Growing the team is something we take joy in.
New **trusted contributors** are nominated by an existing **core developer**,
usually after a stretch of thoughtful, reliable contributions.
A nomination is approved by consensus[^consensus] among the **core developers**,
who are thereby committing to mentor and shepherd new **trusted contributors**
as they grow into the role.

A **trusted contributor** who takes on more work over time is a wonderful candidate
to join the **core developers**, through the very same process.
There is also an alternative path to becoming a **core developer**:
being a member of the NIU who is assigned to develop and maintain `movement`
as part of their job.

Life changes, and so does the time we can give to a project—so you're welcome to step
back at any point by simply letting us know. These roles reflect active involvement,
not a lasting obligation, and stepping away carries no consequences.
The door stays open if you'd like to return.

In the unlikely case that a **core developer** or **trusted contributor** has ceased
to participate in the project for more than three months without notice,
the other **core developers** will reach out to check in. If they don't respond,
the team may decide to revoke or downgrade their access rights.

## Governance model changes

Governance model changes occur through a GitHub pull request that updates
this document. The pull request is marked for review by all **core developers**
and is merged as soon as they all approve, or after two weeks if no objections are raised.

[^consensus]: We use the terms consensus and lazy consensus as defined by the [Apache Software Foundation](https://community.apache.org/committers/decisionMaking.html#lazy-consensus).
