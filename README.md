bestofboth
==========

A tool for smoothing terrain discontinuities in Minecraft worlds.
http://github.com/gmcnew/bestofboth


Introduction
------------
[Minecraft](http://minecraft.net) is a multiplayer adventure game that allows players to dig up the terrain, build structures, explore caves, fight monsters, design circuitry, create portals to [hell](http://www.minecraftwiki.net/wiki/The_Nether), and more. Updates to the game can change its terrain generation code, which causes [unsightly borders](http://www.youtube.com/watch?v=Urhw_kPDkoo) in your world.

Some people dodge this problem by creating a new world whenever Minecraft's terrain code changes. But who wants to lose all of their old creations? With this tool, you can turn those ugly terrain discontinuities into a gentle river and get the **best of both** worlds! *(cue theme music)*


Screenshots
-----------

http://gmcnew.imgur.com/bestofboth


How to use this tool
--------------------

Before doing anything with this program, **back up your world!** There's no "undo" feature, and you might not like the changes it makes to your terrain. You might also have a power outage that causes your world to be corrupted while it's being modified. Either way, better safe than sorry. =)

Also, you should **close Minecraft** before you run this. If you're a server administrator, you should take your world offline.


### Find edges

This script must be run from a command line, and it must be run in two steps. The first is the "--find-edges" step, which examines a world and finds its edges:

    bestofboth --find-edges <path_to_world>

For example:

    bestofboth --find-edges c:\minecraft\saves\world1

This doesn't make any modifications to your world -- it simply finds out where its current boundary is. (The edge data is stored in edges.txt in your world folder.) This boundary represents the dividing line between terrain generated with the *old* terrain generation code and terrain generated with *new* terrain generation code. This dividing line will be smoothed into a river later on.

Next, upgrade Minecraft to a version with new terrain generation code. If you start exploring the boundaries of your world, you should notice large flat walls where things don't line up properly.

(Note: If you run --find-edges and keep exploring your world *before* upgrading Minecraft, your exploring might expand your world's boundaries, causing the old edge data to become out of date. It's up to you to run --find-edges again if this happens, or else the river that's carved in your world will be in the wrong place.)


### Smooth edges

When you're ready to take a break from playing, you can start smoothing out some of those terrain boundaries. (Once again, close Minecraft or stop your server.) You do this with the "--smooth" step, which carves a river between old and new chunks:

    bestofboth --smooth <path_to_world>

This modifies terrain within 16 blocks (1 chunk) of the boundary. If you've built something on that boundary, it will probably be destroyed. Once the boundary has been smoothed, however, you can build on it as much as you like -- it won't be smoothed a second time.

The "--smooth" step will only smooth edges that have been explored. It's possible that many of the giant cliffs in your world haven't been generated yet, simply because nobody has explored those parts of the world and caused new terrain to be generated. In this case, the tool will report something like this:

    79 edge(s) can't be smoothed yet, since they're not fully explored

This is okay! You can do some more exploration whenever you want and then run the "--smooth" step again at a later time.

Eventually, if you run the "--smooth" step after exploring your entire border, bestofboth should report:

    the map is perfectly smoothed -- nothing to do!

At this point a nice, gentle river (with some oceans here and there) should completely encircle your old land.


Summary of typical use
----------------------

    bestofboth --find-edges <path_to_world>
    <upgrade Minecraft to a version with new terrain generation code>
    ...
    <play Minecraft, explore, and cause new terrain to be generated>
    bestofboth --smooth <path_to_world>
    ...
    <more exploration and edge discovery>
    bestofboth --smooth <path_to_world>
    ...
    <finish exploring edges and new terrain along them>
    bestofboth --smooth <path_to_world>


Known issues
------------

See the [list of issues](https://github.com/gmcnew/bestofboth/issues).

These issues only apply to the --smooth step. You should still be able to run --find-edges, explore your world, and wait to run --smooth until you've upgraded to a version of this tool that fixes these issues.

Special thanks
--------------
* [codewarrior0](http://github.com/codewarrior0), whose [pymclevel](http://github.com/codewarrior0/pymclevel) project does about 95% of the hard work for me. =)
