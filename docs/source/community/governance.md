(target-governance)=
# Governance

`movement` is currently led by the [Neuroinformatics Unit](https://neuroinformatics.dev/)
(NIU), a Research Software Engineering team based at the
[Sainsbury Wellcome Centre](https://www.sainsburywellcome.org/web/) and the
[Gatsby Computational Neuroscience Unit](https://www.ucl.ac.uk/gatsby/gatsby-computational-neuroscience-unit).
That said, `movement` is very much a community effort, and we're grateful to
everyone who helps move it forward.

The roles below describe the different ways people can take part and the
repository privileges that comes with each role. To see who currently fills them,
head over to the [](target-people) page.

## Roles

### Contributors

If you've opened a pull request, filed an issue, or joined a discussion through
any of our [community channels](target-connect-with-us),
you're a contributor, and every bit of help counts.
Our [contributing guide](target-contributing) is the place to start.
We strongly believe that contributions directly help us achieve our [mission](target-mission).
The more people get involved, the more robust and versatile `movement` becomes.
Contributors bring fresh perspectives, ensuring that the software serves a broader
and more diverse community of users. Beyond technical improvements,
contributions help us build trust, foster shared ownership,
make the project more sustainable, and strengthen the wider ecosystem of
open-source tools for behavioural analysis.

Everyone who contributes to the GitHub repository—whether through code or documentation—gets
credited on the [](target-people) page. Once your first pull request is merged, a
scheduled monthly workflow will add you to that page, and you'll also get a mention in
the [release notes](movement-github:releases) for that version of `movement`.
Some people contribute in ways that don't involve the repository, e.g. by providing
sample data. We add those contributors to the [](target-people) page manually, so please
let us know if you think we've missed you.

### Team members

Also known as **members** of the `movement` development team,
these are people who have made several contributions to the project and
take on a more active role in its development and maintenance.

**Team members** are welcome to help with a range of tasks, including:

- Keeping the issue tracker and pull requests tidy and welcoming.
- Participating in discussions on [Zulip](movement-zulip:) and attending community calls.
- Reviewing pull requests—especially in parts of the codebase they are familiar with.

On GitHub, this role comes with *Write* access. We trust our **members** to use that judiciously:
merging straightforward changes, particularly in areas they know well,
while seeking input from **maintainers** for larger or more contentious decisions.

There's no expected time commitment and no quota to meet.
Joining the team also comes with a promise from us maintainers:
we will mentor and guide you as you grow into the role.
Some people take on this role for a fixed period—for instance while doing an
[internship](https://neuroinformatics.dev/get-involved/gsoc/index.html) with us,
or while working on `movement` as part of a collaboration.
For some **contributors**, this role is a stepping stone to becoming a **maintainer**.
For others, it is simply a rewarding place to make a lasting contribution.
Both are entirely fine—the role is a valued destination in its own right,
and there is no expectation that you continue beyond it.

:::{note}
This role is inspired by [napari's Triage team](napari:developers/coredev/triage.html),
whose thoughtful approach to community maintenance we've adapted to fit `movement`.
:::

### Maintainers

**Maintainers** are the stewards of `movement`, responsible
for the project's long-term direction and health.
The role's name matches the *Maintain* access it comes with on GitHub,
but you may also see the same group referred to as core developers
or core team.

Beyond everything a **team member** can do, **maintainers** perform the following tasks:

- [Make decisions](#decision-making) about the project's scope and roadmap, and prioritise features and bug fixes. Our community actively shapes these decisions, but **maintainers** make the final call.
- Manage the repository itself, including merging pull requests and making releases.
- Foster the community by moderating conversations on GitHub and Zulip, organising community calls, and posting on social media;
- Publicly represent the project in talks, workshops, and other events. That said, we welcome anyone in our community publicising `movement` in their own way, and we provide a list of [](target-resources) to help you do so.

Administrative access to the GitHub repository, the `movement` package on PyPI,
and our [Zulip chat](movement-zulip:) is reserved for **maintainers** only.
Typically, administrative rights will be held by the
[head of the NIU](https://neuroinformatics.dev/people.html)
and at least one other **maintainer**, to ensure continuity in case of absence.

## Decision-making

Most day-to-day decisions are made by
[lazy consensus](https://community.apache.org/committers/decisionMaking.html#lazy-consensus)
among the **maintainers**. Approval from one **maintainer** is enough to merge most pull requests.
We only escalate to the full team for matters that affect the project's
core architecture, scope, roadmap, or governance.

For these more consequential decisions, we discuss publicly on our
[community channels](target-connect-with-us), and consult all **team members**
and the wider community for input. We aim to reach a
[consensus](https://community.apache.org/committers/decisionMaking.html#consensus)
among **core developers**. In the rare cases we can't,
the [head of the NIU](https://neuroinformatics.dev/people.html)
has the final say and responsibility to decide for the project.

## Joining and stepping down

Growing the team is something we take joy in.
New **team members** are nominated by an existing **maintainer**,
usually after a stretch of thoughtful, reliable contributions.
A nomination is approved by consensus among the **maintainers**, who are thereby
committing to mentor new **team members** as they grow into the role.

A **team member** who takes on more work over time could be well-suited to
become a **maintainer**, should they want to, through the very same process.
There is also an alternative path to becoming a **maintainer**:
being a member of the NIU who is assigned to develop and maintain `movement`
as part of their job.

Life changes, and so does the time we can give to a project—so you're welcome to step
back at any point by simply letting us know. These roles reflect active involvement,
not a lasting obligation, and stepping away carries no consequences.
The door stays open if you'd like to return.

In the unlikely case that a **maintainer** or **team member** has not
exercised their role in any way for longer than three months,
the active **maintainers** will reach out to check in, and may decide
to to revoke or downgrade their access rights, depending on the circumstances.

## Governance model changes

Governance model changes occur through a GitHub pull request that updates
this document. The pull request is marked for review by all **maintainers**
and is merged as soon as they all approve, or after two weeks if no objections are raised.
