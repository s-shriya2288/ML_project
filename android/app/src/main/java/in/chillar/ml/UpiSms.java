package in.chillar.ml;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reads one bank SMS and, if it is a completed UPI/bank debit, pulls out the amount, payee and reference.
 * Plain Java (no Android classes) so it can be tested on a computer. The SMS text itself is never stored or sent.
 */
public final class UpiSms {
    public static final class Debit {
        public final double amount;
        public final String payee;
        public final String ref;

        Debit(double amount, String payee, String ref) {
            this.amount = amount;
            this.payee = payee;
            this.ref = ref;
        }
    }

    private static final Pattern AMOUNT = Pattern.compile("(?i)(?:rs\\.?|inr|₹)\\s*([0-9][0-9,]*(?:\\.[0-9]{1,2})?)");
    private static final Pattern AMOUNT_AFTER_DEBIT = Pattern.compile("(?i)debited\\s+(?:by|for|with)?\\s*(?:rs\\.?|inr|₹)?\\s*([0-9][0-9,]*(?:\\.[0-9]{1,2})?)");
    private static final Pattern DEBIT = Pattern.compile("(?i)\\b(debited|spent|paid|sent|withdrawn|transferred|purchase|txn of)\\b");
    private static final Pattern NOT_A_DEBIT = Pattern.compile("(?i)\\b(otp|one time password|will be debited|to be debited|request|requested|refund|reversed|reversal|failed|declined|unsuccessful|due on|due date|reminder|cashback|autopay set|mandate)\\b");
    private static final Pattern PAYMENT_HINT = Pattern.compile("(?i)\\b(upi|vpa|imps|neft|a/c|acct|account|card)\\b");
    private static final Pattern UPI_PATH = Pattern.compile("(?i)UPI/(?:[A-Z0-9]{2,4}/)?([0-9]{6,})/([A-Za-z0-9 .&@_-]{2,40}?)(?=\\s+(?:not|on|avl|bal)\\b|[.,;/]|$)");  // Axis-style "UPI/P2M/ref/payee"
    private static final Pattern PAYEE_CREDITED = Pattern.compile("(?i);\\s*([A-Za-z0-9 .&@_-]{2,40}?)\\s+credited");
    private static final Pattern PAYEE_TO = Pattern.compile("(?i)\\b(?:paid to|trf to|transfer to|sent to|to vpa|to|at)\\s+([A-Za-z0-9@._& -]{2,40}?)(?=\\s+(?:on|ref|refno|via|upi|avl|avbl|bal|from|thru|using|not)\\b|[.,;(]|$)");
    private static final Pattern REF = Pattern.compile("(?i)(?:upi\\s*ref(?:erence)?(?:\\s*no)?|ref(?:\\s*no)?|refno|rrn|utr|upi)[\\s:.#-]*([0-9]{6,})");

    private UpiSms() {}

    /** Bank and UPI alerts come from short alphanumeric senders like "VM-HDFCBK"; skip messages from people. */
    public static boolean fromBankSender(String address) {
        return address != null && address.matches(".*[A-Za-z].*");
    }

    public static Debit parse(String body) {
        if (body == null) return null;
        String s = body.replaceAll("\\s+", " ").trim();
        if (!DEBIT.matcher(s).find() || NOT_A_DEBIT.matcher(s).find() || !PAYMENT_HINT.matcher(s).find()) return null;
        Matcher m = AMOUNT_AFTER_DEBIT.matcher(s);
        if (!m.find()) {
            m = AMOUNT.matcher(s);
            if (!m.find()) return null;
        }
        double amount;
        try {
            amount = Double.parseDouble(m.group(1).replace(",", ""));
        } catch (NumberFormatException e) {
            return null;
        }
        if (amount <= 0 || amount > 1_000_000) return null;
        String payee = null, ref = null;
        Matcher u = UPI_PATH.matcher(s);
        Matcher p = PAYEE_CREDITED.matcher(s);
        if (u.find()) { ref = u.group(1); payee = u.group(2); }
        else if (p.find()) payee = p.group(1);
        else {
            p = PAYEE_TO.matcher(s);
            while (p.find()) {
                String c = p.group(1).trim();
                if (!c.matches("(?i)(your|you|a/c|acct|account|card|rs|inr).*")) { payee = c; break; }
            }
        }
        if (payee != null) payee = payee.replaceAll("(?i)^vpa\\s+", "").replaceAll("\\s+", " ").trim();
        Matcher r = REF.matcher(s);
        if (ref == null && r.find()) ref = r.group(1);
        return new Debit(amount, payee == null || payee.isEmpty() ? "UPI payment" : payee, ref);
    }
}
