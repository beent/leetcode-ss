using System;
using System.Collections.Generic;
namespace Leetcode
{
    /*
    // Definition for a Node.   
    */
    public class Node
    {
        public int val;
        public IList<Node> children;

        public Node() { }

        public Node(int _val)
        {
            val = _val;
        }

        public Node(int _val, IList<Node> _children)
        {
            val = _val;
            children = _children;
        }
    }

    /**
     * Definition for a binary tree node.
     */
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int x) { val = x; }
    }

    /**
    * Definition for singly-linked list.
    */
    /// <summary>
    /// singly-linked list(单向链表)
    /// </summary>
    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int val = 0, ListNode next = null)
        {
            this.val = val;
            this.next = next;
        }
    }

    public class ListNodeHelper
    {
        /// <summary>
        /// 将链表 l 切掉前 n 个节点，并返回后半部分的链表头
        /// </summary>
        /// <param name="l"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public ListNode Cut(ListNode l, int n)
        {
            ListNode p = l;
            while (--n != 0 && p != null)
            {
                p = p.next;
            }
            if (p == null)
            {
                return null;
            }
            ListNode next = p.next;
            p.next = null;
            return next;
        }

        /// <summary>
        /// 双路归并(ASC)
        /// </summary>
        /// <param name="l1"></param>
        /// <param name="l2"></param>
        /// <returns></returns>
        public ListNode Merge(ListNode l1, ListNode l2)
        {
            ListNode dummyHead = new ListNode(0);
            ListNode p = dummyHead;
            while (l1 != null && l2 != null)
            {
                if (l1.val < l2.val)
                {
                    p.next = l1;
                    p = l1;
                    l1 = l1.next;
                }
                else
                {
                    p.next = l2;
                    p = l2;
                    l2 = l2.next;
                }
            }
            p.next = l1 != null ? l1 : l2;
            return dummyHead.next;
        }


    }

}